from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .time_embedding import TimeEmbedding


class FiLM2d(nn.Module):
    """Channel-wise modulation from fused map + time."""

    def __init__(self, channels: int, cond_ch: int, time_dim: int):
        super().__init__()
        self.norm = _group_norm_safe(channels)
        self.lin = nn.Linear(cond_ch + time_dim, channels * 2)

    def forward(self, x: torch.Tensor, pooled_cond: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        z = torch.cat([pooled_cond, t_emb], dim=1)
        gamma_beta = self.lin(z).unsqueeze(-1).unsqueeze(-1)
        g, b = gamma_beta.chunk(2, dim=1)
        return h * (1.0 + torch.tanh(g)) + b


def _group_norm_safe(num_channels: int) -> nn.GroupNorm:
    g = min(32, num_channels)
    while g > 1 and num_channels % g != 0:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ConvGnAct(nn.Sequential):
    def __init__(self, ci: int, co: int, k=3, s=1, p=None):
        if p is None:
            p = k // 2
        super().__init__(
            nn.Conv2d(ci, co, k, s, p, bias=False),
            _group_norm_safe(co),
            nn.ReLU(inplace=True),
        )


class VelocityConditionalUNet(nn.Module):
    """
    Hourglass predicts velocity field v_theta(x,t) at latent resolution (/4 of input mask),
    then upsamples to mask resolution with last conv.
    Conditioning: FiLM derived from pooled fused pyramid features matching each decoder scale.
    """

    def __init__(self, fuse_channels: tuple[int, int, int] = (48, 96, 144), hidden: int = 80, time_dim: int = 128):
        super().__init__()
        c0, c1, c2 = fuse_channels
        self.time_emb = TimeEmbedding(time_dim)

        cond_vec = c0 + c1 + c2
        pooled_dim_per_level = [c0, c1, c2]

        self.stem_lat = ConvGnAct(1, hidden // 2, k=7, s=4, p=3)
        self.stem_joint = ConvGnAct(hidden // 2 + c0, hidden, k=1)

        self.film0 = FiLM2d(hidden, pooled_dim_per_level[0], time_dim)
        self.down1 = ConvGnAct(hidden, hidden * 2, k=3, s=2)
        self.film1 = FiLM2d(hidden * 2, pooled_dim_per_level[1], time_dim)
        self.down2 = ConvGnAct(hidden * 2, hidden * 2, k=3, s=2)
        self.mid_film = FiLM2d(hidden * 2, pooled_dim_per_level[2], time_dim)
        self.mid_conv1 = ConvGnAct(hidden * 2, hidden * 2, k=3, s=1)
        self.mid_conv2 = ConvGnAct(hidden * 2, hidden * 2, k=3, s=1)

        self.up1 = ConvGnAct(hidden * 2 + hidden * 2, hidden * 2, k=3)
        self.film_dec1 = FiLM2d(hidden * 2, pooled_dim_per_level[1], time_dim)

        self.up2 = ConvGnAct(hidden * 2 + hidden, hidden, k=3)
        self.film_dec0 = FiLM2d(hidden, pooled_dim_per_level[0], time_dim)

        self.out_conv = nn.Conv2d(hidden, 1, kernel_size=1)

        self.fc_cond_global = nn.Linear(cond_vec, hidden)

    def _pool(self, feat: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(feat, 1).flatten(1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, fused_feats: List[torch.Tensor]) -> torch.Tensor:
        """x_t [B,1,H,W]; t [B]; fused_feats [F0,F1,F2] at resolutions H/4,H/8,H/16 of input."""
        f0, f1, f2 = fused_feats
        t_emb = self.time_emb(t)
        pc0 = self._pool(f0)
        pc1 = self._pool(f1)
        pc2 = self._pool(f2)
        p_all = torch.cat([pc0, pc1, pc2], dim=1)
        aux_vec = torch.relu(self.fc_cond_global(p_all))

        xl = self.stem_lat(x_t)
        if xl.shape[-2:] != f0.shape[-2:]:
            xl = F.interpolate(xl, size=f0.shape[-2:], mode="bilinear", align_corners=False)

        xl = torch.cat([xl, f0], dim=1)
        h0 = self.stem_joint(xl)
        h0 = self.film0(h0, pc0, t_emb) + aux_vec.unsqueeze(-1).unsqueeze(-1) * 0.05

        h1 = self.down1(h0)
        if h1.shape[-2:] != f1.shape[-2:]:
            h1 = F.interpolate(h1, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        h1 = self.film1(h1, pc1, t_emb)

        h2 = self.down2(h1)
        if h2.shape[-2:] != f2.shape[-2:]:
            h2 = F.interpolate(h2, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        h_mid = self.mid_film(h2, pc2, t_emb)
        h_mid = self.mid_conv1(h_mid)
        h_mid = self.mid_conv2(h_mid)

        u1 = F.interpolate(h_mid, scale_factor=2, mode="bilinear", align_corners=False)
        cat1 = torch.cat([u1, h1], dim=1)
        h_dec1 = self.up1(cat1)
        if h_dec1.shape[-2:] != f1.shape[-2:]:
            h_dec1 = F.interpolate(h_dec1, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        h_dec1 = self.film_dec1(h_dec1, pc1, t_emb)

        u0 = F.interpolate(h_dec1, scale_factor=2, mode="bilinear", align_corners=False)
        cat0 = torch.cat([u0, h0], dim=1)
        h_dec0 = self.up2(cat0)
        if h_dec0.shape[-2:] != f0.shape[-2:]:
            h_dec0 = F.interpolate(h_dec0, size=f0.shape[-2:], mode="bilinear", align_corners=False)
        h_dec0 = self.film_dec0(h_dec0, pc0, t_emb)

        v_lr = self.out_conv(h_dec0)

        hi = x_t.shape[-1]
        if v_lr.shape[-2:] != (hi, hi):
            v_hi = F.interpolate(v_lr, size=(hi, hi), mode="bilinear", align_corners=False)
        else:
            v_hi = v_lr

        assert v_hi.shape[-2:] == x_t.shape[-2:]
        return v_hi
