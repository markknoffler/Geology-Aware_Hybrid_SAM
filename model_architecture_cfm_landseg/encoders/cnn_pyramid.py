from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sobel_mag(x: torch.Tensor) -> torch.Tensor:
    """Finite-difference slope magnitude for DEM channel [B,1,H,W]."""
    gx = x[:, :, :, 1:] - x[:, :, :, :-1]
    gx = F.pad(gx, (0, 1, 0, 0), mode="replicate")
    gy = x[:, :, 1:, :] - x[:, :, :-1, :]
    gy = F.pad(gy, (0, 0, 0, 1), mode="replicate")
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return F.relu(x + h)


class PyramidEncoder(nn.Module):
    """Three-scale CNN pyramid: strides produce H/4, H/8, H/16 (for 256-input)."""

    def __init__(self, in_channels: int, width: int = 48, dem_branch: bool = False):
        super().__init__()
        self.dem_branch = dem_branch
        stem_in = in_channels + 1 if dem_branch else in_channels

        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, width, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.block0 = nn.Sequential(ResidualBlock(width), ResidualBlock(width))
        self.down1 = nn.Sequential(
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
        )
        self.block1 = nn.Sequential(ResidualBlock(width * 2), ResidualBlock(width * 2))
        self.down2 = nn.Sequential(
            nn.Conv2d(width * 2, width * 3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width * 3),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(ResidualBlock(width * 3), ResidualBlock(width * 3))
        self._out_channels: Tuple[int, int, int] = (width, width * 2, width * 3)

    @property
    def out_channels(self) -> Tuple[int, int, int]:
        return self._out_channels

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.dem_branch and x.shape[1] == 1:
            slope = _sobel_mag(x)
            x = torch.cat([x, slope], dim=1)
        h0 = self.stem(x)
        h0 = self.block0(h0)
        h1 = self.down1(h0)
        h1 = self.block1(h1)
        h2 = self.down2(h1)
        h2 = self.block2(h2)
        return [h0, h1, h2]
