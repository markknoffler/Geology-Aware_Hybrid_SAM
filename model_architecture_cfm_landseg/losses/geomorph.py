"""DEM-aligned geomorphological regularization."""

from __future__ import annotations

import torch


def geometric_gradients(tensor: torch.Tensor):
    gx = tensor[:, :, :, 1:] - tensor[:, :, :, :-1]
    gx = torch.nn.functional.pad(gx, (0, 1, 0, 0), mode="replicate")
    gy = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    gy = torch.nn.functional.pad(gy, (0, 0, 0, 1), mode="replicate")
    return gx, gy


def geomorph_alignment_loss(pred_logits: torch.Tensor, dem_1ch: torch.Tensor, mask_weight_eps: float = 1e-6) -> torch.Tensor:
    """
    Encourage segmentation boundaries to concentrate where terrain slope magnitude is larger.
    penalize |(grad m)| weighted by downward alignment with topography (soft variant).

    m: sigmoid(logits).
    Uses dot(grad m, grad H) clipped: penalize large mask gradient on flat DEM (inverse weight by slope magnitude).
    L = mean( |grad m|^2 / (alpha + slope_mag) )

    DEM must be differentiable-same-device tensor [B,1,H,W] (normalized).
    """
    m = torch.sigmoid(pred_logits)
    gx_m, gy_m = geometric_gradients(m)
    gxm_h, gym_h = geometric_gradients(dem_1ch)
    slope = torch.sqrt(gxm_h ** 2 + gym_h ** 2 + mask_weight_eps)
    gm2 = gx_m ** 2 + gy_m ** 2
    # Boundary energy reduced on slopes; flat regions discourage spurious fragmentation
    w = 1.0 / (1.0 + 5.0 * slope / (slope.mean(dim=(2, 3), keepdim=True).detach() + mask_weight_eps))
    return (gm2 * w).mean()

