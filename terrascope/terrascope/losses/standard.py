from typing import Optional

import torch
import torch.nn.functional as F


def bce_with_logits_loss(
    logits: torch.Tensor, target: torch.Tensor, pos_weight: Optional[torch.Tensor] = None
):
    return F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=pos_weight)


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    p = torch.sigmoid(logits)
    t = target.float()
    inter = (p * t).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + t.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (union + eps)
    return 1 - dice.mean()


def tversky_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.7, beta: float = 0.3, eps: float = 1e-6):
    p = torch.sigmoid(logits)
    t = target.float()
    tp = (p * t).sum(dim=(2, 3))
    fp = (p * (1 - t)).sum(dim=(2, 3))
    fn = ((1 - p) * t).sum(dim=(2, 3))
    ti = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1 - ti.mean()


def focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25, eps: float = 1e-6):
    p = torch.sigmoid(logits)
    t = target.float()
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    p_t = p * t + (1 - p) * (1 - t)
    mod = (1 - p_t).pow(gamma)
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    return (alpha_t * mod * ce).mean()


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    p = torch.sigmoid(logits)
    t = target.float()
    inter = (p * t).sum(dim=(2, 3))
    union = p.sum(dim=(2, 3)) + t.sum(dim=(2, 3)) - inter
    iou = (inter + eps) / (union + eps)
    return 1 - iou.mean()


def gradient_l1_boundary_loss(logits: torch.Tensor, target: torch.Tensor):
    """Match spatial gradients of probability map to mask edges (no extra deps)."""
    p = torch.sigmoid(logits)
    t = target.float()

    def sobel(x):
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    px, py = sobel(p)
    tx, ty = sobel(t)
    px = F.pad(px, (0, 1, 0, 0))
    py = F.pad(py, (0, 0, 0, 1))
    tx = F.pad(tx, (0, 1, 0, 0))
    ty = F.pad(ty, (0, 0, 0, 1))
    return (px - tx).abs().mean() + (py - ty).abs().mean()
