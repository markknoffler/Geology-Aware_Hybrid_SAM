from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from .geomorph import geomorph_alignment_loss
from .tversky_module import TverskyLoss, resize_target_like


def mask_to_latent_z(y: torch.Tensor, sigma: float = 4.0) -> torch.Tensor:
    """Map binary mask probabilities to latent z for flow matching."""
    eps = 1e-3
    p = torch.clamp(y.float(), eps, 1.0 - eps)
    centered = torch.logit(p)
    return sigma * centered


class TriCFMCompositeLoss(nn.Module):
    """Loss on model.output_dict from TriEncoderCFMNet."""

    def __init__(
        self,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        fm_weight: float = 1.0,
        seg_weight: float = 2.0,
        geo_weight: float = 0.15,
        vsmooth_weight: float = 0.05,
        latent_sigma: float = 4.0,
        fm_residual_scale_sq: Optional[float] = None,
    ):
        super().__init__()
        self.fm_weight = fm_weight
        self.seg_weight = seg_weight
        self.geo_weight = geo_weight
        self.vsmooth_weight = vsmooth_weight
        self.latent_sigma = latent_sigma
        # Target velocity is ε − z with |z| ≈ σ × |logit(p)| (~6–7 typical after clipping).
        self.fm_residual_scale_sq = fm_residual_scale_sq or max(
            1.0, (latent_sigma * 6.0) ** 2
        )
        self.tversky = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)

    def forward(self, outputs: Any, target: torch.Tensor, dem_chw: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            outputs: dict with logits_aux, optionally fm_residual, velocities, epsilon, latent_z...
            dem_chw [B,1,H,W]: single DEM band for Bijie/L4S; if None skips geo branch.
        """
        if isinstance(outputs, (tuple, list)):
            logits = outputs[0]
            out = {}
        elif isinstance(outputs, Mapping):
            out = outputs
            logits = out["logits_aux"]
        else:
            logits = outputs
            out = {}

        t_main = resize_target_like(target, logits)
        loss = self.seg_weight * self.tversky(logits, t_main)

        if "fm_residual" in out and self.fm_weight > 0:
            fm_mean = out["fm_residual"].mean()
            loss = loss + self.fm_weight * (fm_mean / self.fm_residual_scale_sq)

        if self.vsmooth_weight > 0 and "v_smooth_penalty" in out:
            loss = loss + self.vsmooth_weight * (
                out["v_smooth_penalty"] / self.fm_residual_scale_sq
            )

        if self.geo_weight > 0 and dem_chw is not None:
            loss = loss + self.geo_weight * geomorph_alignment_loss(logits, dem_chw.detach())

        return loss


def compute_fm_residuals_batch(
    v_pred: torch.Tensor,
    epsilon: torch.Tensor,
    latent_z: torch.Tensor,
) -> torch.Tensor:
    """OT straight-path conditional FM target residual."""
    tgt = epsilon - latent_z
    return (v_pred - tgt) ** 2


