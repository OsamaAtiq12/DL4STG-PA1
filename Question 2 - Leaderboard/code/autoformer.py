import math

import torch
from torch import nn
from torch.nn import functional as F


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel=25):
        super().__init__()
        if kernel < 1 or kernel % 2 == 0:
            raise ValueError("kernel must be positive and odd")
        self.kernel = kernel

    def forward(self, x):
        pad = self.kernel // 2
        x_t = x.transpose(1, 2)
        padded = F.pad(x_t, (pad, pad), mode="replicate")
        trend = F.avg_pool1d(padded, kernel_size=self.kernel, stride=1).transpose(1, 2)
        return x - trend, trend


class AutoCorrelation(nn.Module):
    def __init__(self, d_model, n_heads, factor=1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.factor = factor
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, queries, keys, values):
        b, lq, d = queries.shape
        lk = keys.shape[1]
        h = self.n_heads
        dh = d // h

        q = self.query(queries).view(b, lq, h, dh).permute(0, 2, 3, 1)
        k = self.key(keys).view(b, lk, h, dh).permute(0, 2, 3, 1)
        v = self.value(values).view(b, lk, h, dh).permute(0, 2, 3, 1)

        length = max(lq, lk)
        if k.shape[-1] < length:
            k = F.pad(k, (0, length - k.shape[-1]))
            v = F.pad(v, (0, length - v.shape[-1]))
        if q.shape[-1] < length:
            q = F.pad(q, (0, length - q.shape[-1]))

        q_fft = torch.fft.rfft(q.float(), dim=-1)
        k_fft = torch.fft.rfft(k.float(), dim=-1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=length, dim=-1)
        corr = corr.mean(dim=2)

        top_k = max(1, min(length, self.factor * int(math.ceil(math.log(max(length, 2))))))
        weights, delays = torch.topk(corr, top_k, dim=-1)
        weights = torch.softmax(weights, dim=-1)

        t = torch.arange(length, device=v.device)
        src = (t.view(1, 1, 1, length) - delays.unsqueeze(-1)) % length
        src = src.long().unsqueeze(2).expand(b, h, dh, top_k, length)
        gathered = torch.gather(v.unsqueeze(3).expand(b, h, dh, top_k, length), -1, src)
        mixed = (gathered * weights.unsqueeze(2).unsqueeze(-1)).sum(dim=3)
        mixed = mixed[..., :lq].permute(0, 3, 1, 2).contiguous().view(b, lq, d)
        return self.out(mixed)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, moving_avg, dropout, factor):
        super().__init__()
        self.corr = AutoCorrelation(d_model, n_heads, factor)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        y = self.norm1(x)
        x = x + self.drop(self.corr(y, y, y))
        seasonal, _ = self.decomp1(x)
        seasonal = seasonal + self.drop(self.ff(self.norm2(seasonal)))
        seasonal, _ = self.decomp2(seasonal)
        return seasonal


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, moving_avg, dropout, factor):
        super().__init__()
        self.self_corr = AutoCorrelation(d_model, n_heads, factor)
        self.cross_corr = AutoCorrelation(d_model, n_heads, factor)
        self.decomp1 = SeriesDecomposition(moving_avg)
        self.decomp2 = SeriesDecomposition(moving_avg)
        self.decomp3 = SeriesDecomposition(moving_avg)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.trend_proj = nn.Linear(d_model, d_model)

    def forward(self, x, cross):
        y = self.norm1(x)
        x = x + self.drop(self.self_corr(y, y, y))
        seasonal, trend1 = self.decomp1(x)

        c = self.norm2(cross)
        seasonal = seasonal + self.drop(self.cross_corr(self.norm2(seasonal), c, c))
        seasonal, trend2 = self.decomp2(seasonal)

        seasonal = seasonal + self.drop(self.ff(self.norm3(seasonal)))
        seasonal, trend3 = self.decomp3(seasonal)

        trend = self.trend_proj(trend1 + trend2 + trend3)
        return seasonal, trend


class Autoformer(nn.Module):
    def __init__(
        self,
        enc_in,
        dec_in,
        c_out=1,
        seq_len=336,
        label_len=168,
        pred_len=168,
        d_model=32,
        n_heads=4,
        e_layers=1,
        d_layers=1,
        d_ff=64,
        moving_avg=25,
        dropout=0.1,
        factor=1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.decomp = SeriesDecomposition(moving_avg)
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.dec_embedding = nn.Linear(dec_in, d_model)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, moving_avg, dropout, factor) for _ in range(e_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, d_ff, moving_avg, dropout, factor) for _ in range(d_layers)]
        )
        self.projection = nn.Linear(d_model, c_out)
        self.trend_projection = nn.Linear(d_model, c_out)

    def forward(self, x_enc, x_dec):
        enc = self.enc_embedding(x_enc)
        for layer in self.encoder:
            enc = layer(enc)

        _, trend_init = self.decomp(x_enc)
        dec = self.dec_embedding(x_dec)
        trend_part = self.enc_embedding(trend_init).mean(dim=1, keepdim=True).repeat(1, x_dec.shape[1], 1)
        seasonal_part = dec

        for layer in self.decoder:
            seasonal_part, trend_delta = layer(seasonal_part, enc)
            trend_part = trend_part + trend_delta

        out = self.projection(seasonal_part) + self.trend_projection(trend_part)
        return out[:, -self.pred_len :, :]


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
