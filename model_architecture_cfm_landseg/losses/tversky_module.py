import torch
import torch.nn as nn
import torch.nn.functional as F


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)
        target = target.float()
        probs = probs.reshape(-1)
        target = target.reshape(-1)

        tp = (probs * target).sum()
        fp = ((1.0 - target) * probs).sum()
        fn = (target * (1.0 - probs)).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


def resize_target_like(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    if target.shape[-2:] != pred.shape[-2:]:
        return F.interpolate(target.float(), size=pred.shape[-2:], mode="nearest")
    return target
