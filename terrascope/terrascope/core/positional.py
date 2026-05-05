import torch
import torch.nn as nn


class PEG(nn.Module):
    """Position encoding generator via depthwise conv (PEG, CVPR-style)."""

    def __init__(self, dim: int, k: int = 3):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, k, padding=k // 2, groups=dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        b, n, c = tokens.shape
        x = tokens.transpose(1, 2).reshape(b, c, H, W)
        pos = self.dw_conv(x)
        x = x + self.gamma * pos
        return x.flatten(2).transpose(1, 2)
