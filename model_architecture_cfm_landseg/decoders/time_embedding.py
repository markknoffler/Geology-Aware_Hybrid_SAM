from __future__ import annotations

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """Lightweight MLP on scalar time for conditioning."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(1, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.reshape(1)
        t = t.reshape(-1, 1).float()
        return self.fc(t)

