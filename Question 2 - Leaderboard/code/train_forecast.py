from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from autoformer import Autoformer, count_trainable_params

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data"
OUT = ROOT / "outputs"
PRED_LEN = 168


@dataclass
class Config:
    seq_len: int = 336
    label_len: int = 84
    pred_len: int = PRED_LEN
    d_model: int = 32
    n_heads: int = 4
    e_layers: int = 1
    d_layers: int = 1
    d_ff: int = 64
    moving_avg: int = 25
    dropout: float = 0.1
    factor: int = 1
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 8
    stride: int = 24
    use_external: bool = False
    val_horizon_blocks: int = 4
    device: str = "cpu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_frames(use_external: bool):
    train = pd.read_csv(DATA / "student_train.csv")
    test = pd.read_csv(DATA / "student_test.csv")
    y = train["value"].to_numpy(dtype=np.float32)
    if use_external:
        ext = pd.read_csv(DATA / "optional_external_data.csv")
        feat_cols = [c for c in ext.columns if c != "time_idx"]
        X = ext[feat_cols].to_numpy(dtype=np.float32)
        mu = X[: len(y)].mean(axis=0, keepdims=True)
        sd = X[: len(y)].std(axis=0, keepdims=True) + 1e-6
        X = (X - mu) / sd
    else:
        X = None
        feat_cols = []
    return y, X, feat_cols, test


class WindowDataset(Dataset):
    def __init__(self, y, X, origins, seq_len, label_len, pred_len):
        self.y = y
        self.X = X
        self.origins = origins
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.origins)

    def __getitem__(self, idx):
        t = self.origins[idx]
        s = t - self.seq_len
        e = t + self.pred_len
        y_hist = self.y[s:t]
        y_future = self.y[t:e]

        if self.X is None:
            x_enc = y_hist[:, None]
            dec_y = np.concatenate(
                [self.y[t - self.label_len : t], np.zeros(self.pred_len, dtype=np.float32)], axis=0
            )[:, None]
            x_dec = dec_y
        else:
            x_enc = np.concatenate([y_hist[:, None], self.X[s:t]], axis=1)
            dec_y = np.concatenate(
                [self.y[t - self.label_len : t], np.zeros(self.pred_len, dtype=np.float32)], axis=0
            )[:, None]
            x_dec = np.concatenate(
                [dec_y, self.X[t - self.label_len : t + self.pred_len]], axis=1
            )

        return (
            torch.from_numpy(x_enc),
            torch.from_numpy(x_dec),
            torch.from_numpy(y_future[:, None]),
        )


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
    denom = np.abs(y_true) + np.abs(y_pred)
    smape = 100.0 * np.mean(np.where(denom < 1e-8, 0.0, 2.0 * np.abs(y_pred - y_true) / denom))
    return {"MAE": mae, "RMSE": rmse, "sMAPE": smape}


def make_origins(n_hist, seq_len, pred_len, stride, start, end):
    lo = max(seq_len, start)
    hi = min(n_hist - pred_len, end)
    return list(range(lo, hi + 1, stride))


def train_one(cfg: Config, seed: int, y_scaled, X, scale_mu, scale_sd):
    set_seed(seed)
    device = torch.device(cfg.device)
    n = len(y_scaled)
    val_len = cfg.val_horizon_blocks * cfg.pred_len
    val_start = n - val_len
    train_origins = make_origins(
        n, cfg.seq_len, cfg.pred_len, cfg.stride, cfg.seq_len, val_start - cfg.pred_len
    )
    val_origins = list(range(val_start, n - cfg.pred_len + 1, cfg.pred_len))

    enc_in = 1 if X is None else (1 + X.shape[1])
    model = Autoformer(
        enc_in=enc_in,
        dec_in=enc_in,
        c_out=1,
        seq_len=cfg.seq_len,
        label_len=cfg.label_len,
        pred_len=cfg.pred_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        e_layers=cfg.e_layers,
        d_layers=cfg.d_layers,
        d_ff=cfg.d_ff,
        moving_avg=cfg.moving_avg,
        dropout=cfg.dropout,
        factor=cfg.factor,
    ).to(device)

    train_loader = DataLoader(
        WindowDataset(y_scaled, X, train_origins, cfg.seq_len, cfg.label_len, cfg.pred_len),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        WindowDataset(y_scaled, X, val_origins, cfg.seq_len, cfg.label_len, cfg.pred_len),
        batch_size=max(1, min(8, len(val_origins))),
        shuffle=False,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_state = None
    best_rmse = float("inf")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total = 0.0
        nobs = 0
        for x_enc, x_dec, yb in train_loader:
            x_enc, x_dec, yb = x_enc.to(device), x_dec.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(x_enc, x_dec)
            loss = loss_fn(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * len(yb)
            nobs += len(yb)

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x_enc, x_dec, yb in val_loader:
                pred = model(x_enc.to(device), x_dec.to(device)).cpu().numpy()
                preds.append(pred)
                trues.append(yb.numpy())
        pred_u = np.concatenate(preds, axis=0) * scale_sd + scale_mu
        true_u = np.concatenate(trues, axis=0) * scale_sd + scale_mu
        m = metrics(true_u, pred_u)
        history.append({"epoch": epoch, "train_mse": total / max(nobs, 1), **m})
        print(
            f"seed={seed} epoch={epoch}/{cfg.epochs} "
            f"val_RMSE={m['RMSE']:.4f} MAE={m['MAE']:.4f} sMAPE={m['sMAPE']:.2f}",
            flush=True,
        )
        if m["RMSE"] < best_rmse:
            best_rmse = m["RMSE"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, history, best_rmse


@torch.no_grad()
def forecast_future(model, y_scaled, X, cfg: Config, scale_mu, scale_sd):
    device = next(model.parameters()).device
    t = len(y_scaled)
    s = t - cfg.seq_len
    y_hist = y_scaled[s:t]
    if X is None:
        x_enc = torch.from_numpy(y_hist[:, None][None, ...]).to(device)
        dec_y = np.concatenate(
            [y_scaled[t - cfg.label_len : t], np.zeros(cfg.pred_len, dtype=np.float32)]
        )[:, None]
        x_dec = torch.from_numpy(dec_y[None, ...]).to(device)
    else:
        x_enc = torch.from_numpy(
            np.concatenate([y_hist[:, None], X[s:t]], axis=1)[None, ...]
        ).to(device)
        dec_y = np.concatenate(
            [y_scaled[t - cfg.label_len : t], np.zeros(cfg.pred_len, dtype=np.float32)]
        )[:, None]
        x_dec = torch.from_numpy(
            np.concatenate([dec_y, X[t - cfg.label_len : t + cfg.pred_len]], axis=1)[None, ...]
        ).to(device)
    model.eval()
    pred = model(x_enc, x_dec).cpu().numpy().reshape(-1)
    pred = pred * scale_sd + scale_mu
    return np.clip(pred, 0.0, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_external", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--seq_len", type=int, default=336)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=48)
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        seq_len=args.seq_len,
        epochs=args.epochs,
        d_model=args.d_model,
        batch_size=args.batch_size,
        stride=args.stride,
        use_external=bool(args.use_external),
        device="cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu",
    )

    y_raw, X, feat_cols, test = load_frames(cfg.use_external)
    n = len(y_raw)
    val_len = cfg.val_horizon_blocks * cfg.pred_len
    train_region = y_raw[: n - val_len]
    mu = float(train_region.mean())
    sd = float(train_region.std() + 1e-6)
    y_scaled = ((y_raw - mu) / sd).astype(np.float32)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    rows = []
    models = []
    for seed in seeds:
        model, history, best_rmse = train_one(cfg, seed, y_scaled, X, mu, sd)
        models.append(model)
        for h in history:
            rows.append({"seed": seed, "use_external": int(cfg.use_external), **h})
        print(f"BEST seed={seed} val_RMSE={best_rmse:.4f} params={count_trainable_params(model)}", flush=True)

    pd.DataFrame(rows).to_csv(OUT / f"val_history_ext{int(cfg.use_external)}.csv", index=False)

    if args.final:
        mu_f = float(y_raw.mean())
        sd_f = float(y_raw.std() + 1e-6)
        y_full = ((y_raw - mu_f) / sd_f).astype(np.float32)
        preds = []
        total_epochs = 0
        total_params = 0
        for seed in seeds:
            cfg_full = Config(**{**asdict(cfg), "val_horizon_blocks": 1, "epochs": max(3, cfg.epochs // 2)})
            model, history, best_rmse = train_one(cfg_full, seed, y_full, X, mu_f, sd_f)
            preds.append(forecast_future(model, y_full, X, cfg_full, mu_f, sd_f))
            total_epochs += cfg_full.epochs
            total_params += count_trainable_params(model)
            print(f"FINAL seed={seed} holdout_RMSE={best_rmse:.4f}", flush=True)

        pred_mean = np.mean(np.stack(preds, axis=0), axis=0)
        (OUT / "predictions.txt").write_text(
            ",".join(f"{v:.6f}" for v in pred_mean.tolist()) + "\n", encoding="utf-8"
        )
        pd.DataFrame({"time_idx": test["time_idx"], "value": pred_mean}).to_csv(
            OUT / "student_test_filled.csv", index=False
        )
        meta = {"P": int(total_params), "E": int(total_epochs), "use_external": int(cfg.use_external)}
        (OUT / "submission_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print("P=", meta["P"], "E=", meta["E"])


if __name__ == "__main__":
    main()
