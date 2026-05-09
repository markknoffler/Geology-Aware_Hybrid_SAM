from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from . import terrain_multistream_losses as terrain_losses
from . import standard as standard_mod


@dataclass
class LossWeights:
    bce: float = 0.0
    dice: float = 0.0
    tversky: float = 1.0
    focal: float = 0.0
    soft_iou: float = 0.0
    boundary: float = 0.0
    tgbc: float = 0.0
    cscd: float = 0.0
    tversky_alpha: float = 0.7
    tversky_beta: float = 0.3
    bce_pos_weight: float = 40.0


def composite_segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dem: torch.Tensor,
    aux: Optional[Tuple[torch.Tensor, torch.Tensor]],
    w: LossWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    parts: dict[str, torch.Tensor] = {}
    if w.bce > 0:
        pos_weight = torch.tensor([w.bce_pos_weight], device=logits.device)
        parts["bce"] = standard_mod.bce_with_logits_loss(logits, target, pos_weight=pos_weight)
    if w.dice > 0:
        parts["dice"] = standard_mod.dice_loss(logits, target)
    if w.tversky > 0:
        parts["tversky"] = standard_mod.tversky_loss(
            logits, target, alpha=w.tversky_alpha, beta=w.tversky_beta
        )
    if w.focal > 0:
        parts["focal"] = standard_mod.focal_loss(logits, target)
    if w.soft_iou > 0:
        parts["soft_iou"] = standard_mod.soft_iou_loss(logits, target)
    if w.boundary > 0:
        parts["boundary"] = standard_mod.gradient_l1_boundary_loss(logits, target)
    if w.tgbc > 0:
        dem_i = dem if dem.shape[-2:] == logits.shape[-2:] else F.interpolate(
            dem, logits.shape[-2:], mode="bilinear", align_corners=False
        )
        parts["tgbc"] = terrain_losses.tgbc_loss(logits, dem_i)
    if w.cscd > 0 and aux is not None:
        parts["cscd"] = terrain_losses.cscd_loss(logits, aux[0], aux[1])

    weights = {
        "bce": w.bce,
        "dice": w.dice,
        "tversky": w.tversky,
        "focal": w.focal,
        "soft_iou": w.soft_iou,
        "boundary": w.boundary,
        "tgbc": w.tgbc,
        "cscd": w.cscd,
    }
    if not parts:
        raise ValueError("No loss terms active: set at least one loss weight > 0.")
    total = torch.zeros((), device=logits.device, dtype=logits.dtype)
    for k, v in parts.items():
        total = total + weights[k] * v

    log = {k: float(v.detach()) for k, v in parts.items()}
    log["total"] = float(total.detach())
    return total, log
