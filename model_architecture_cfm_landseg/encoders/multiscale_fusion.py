from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrimodalGatedFusion(nn.Module):
    """
    Per scale: softmax channel gates (global summary) mixing three tensors of identical shape,
    then spatial modulation γ in [0,1] from depthwise path (PMCNet-style dynamic fusion).
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.gate_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 3, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )
        self.spatial_gamma = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False),
            nn.Sigmoid(),
        )
        self.out_norm = nn.BatchNorm2d(channels)

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([a, b, c], dim=1)
        logits = self.gate_mlp(cat).view(cat.shape[0], 3, 1, 1)
        weights = torch.softmax(logits, dim=1)
        w0, w1, w2 = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        mixed = w0 * a + w1 * b + w2 * c
        gamma = self.spatial_gamma(cat)
        out = gamma * mixed + (1.0 - gamma) * ((a + b + c) / 3.0)
        out = self.out_norm(out)
        return out


class MultiScaleFusionStack(nn.Module):
    """Fusion module applied independently at each pyramid level."""

    def __init__(self, channels_per_level: List[int]):
        super().__init__()
        self.blocks = nn.ModuleList([TrimodalGatedFusion(c) for c in channels_per_level])

    def forward(
        self, feats_a: List[torch.Tensor], feats_b: List[torch.Tensor], feats_c: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        fused = []
        for i, fusion in enumerate(self.blocks):
            fused.append(fusion(feats_a[i], feats_b[i], feats_c[i]))
        return fused
