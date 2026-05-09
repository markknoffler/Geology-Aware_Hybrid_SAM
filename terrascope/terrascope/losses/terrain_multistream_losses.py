"""
Novel objectives for dual-stream landslide segmentation.

TGBC — Topographic gradient–boundary calibration: align soft mask boundaries with DEM slope field.
CSCD — Cross-stream calibrated disagreement: selective symmetric KL on uncertain pixels between auxiliary heads.
"""

import torch
import torch.nn.functional as F


def dem_sobel_unit(dem: torch.Tensor, eps: float = 1e-6):
    """Unit 2D terrain gradient from single-channel DEM (B,1,H,W)."""
    gx = dem[:, :, :, 1:] - dem[:, :, :, :-1]
    gy = dem[:, :, 1:, :] - dem[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    g = torch.cat([gx, gy], dim=1)
    n = g.norm(dim=1, keepdim=True).clamp_min(eps)
    return g / n


def mask_gradient(prob: torch.Tensor):
    gx = prob[:, :, :, 1:] - prob[:, :, :, :-1]
    gy = prob[:, :, 1:, :] - prob[:, :, :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.cat([gx, gy], dim=1)


def tgbc_loss(
    logits: torch.Tensor,
    dem: torch.Tensor,
    band_percentile: float = 0.85,
    ortho_weight: float = 0.15,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Topographic gradient–boundary calibration (TGBC).

    Let g be the unit DEM gradient and m = |∇σ(logits)| a soft boundary strength.
    We encourage cosine(g, ∇p) on pixels in the top boundary band, and mild orthogonality elsewhere
    (ridge-running edges vs along-contour structure).
    """
    p = torch.sigmoid(logits)
    # Landslide4Sense has 3 topo channels; average them to a single surface for gradient alignment.
    dem_single = dem.mean(dim=1, keepdim=True) if dem.size(1) > 1 else dem
    g = dem_sobel_unit(dem_single)
    mp = mask_gradient(p)
    mp_n = mp.norm(dim=1, keepdim=True).clamp_min(eps)
    mp_u = mp / mp_n

    flat = mp_n.view(mp_n.size(0), -1)
    thresh = torch.quantile(flat, band_percentile, dim=1, keepdim=True).view(-1, 1, 1, 1)
    band = (mp_n >= thresh).float()

    cos = (g * mp_u).sum(dim=1, keepdim=True).clamp(-1, 1)
    align = (1.0 - cos) * band
    ortho = (g * mp_u).sum(dim=1).abs() * (1.0 - band.squeeze(1))
    return align.mean() + ortho_weight * ortho.mean()


def symmetric_kl_logits(log_a: torch.Tensor, log_b: torch.Tensor, eps: float = 1e-6):
    pa = torch.sigmoid(log_a).clamp(eps, 1 - eps)
    pb = torch.sigmoid(log_b).clamp(eps, 1 - eps)
    kl_ab = pa * torch.log(pa / pb) + (1 - pa) * torch.log((1 - pa) / (1 - pb))
    kl_ba = pb * torch.log(pb / pa) + (1 - pb) * torch.log((1 - pb) / (1 - pa))
    return 0.5 * (kl_ab + kl_ba)


def cscd_loss(
    logits_main: torch.Tensor,
    logits_rgb: torch.Tensor,
    logits_dem: torch.Tensor,
    uncertainty_quantile: float = 0.75,
    agree_weight: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Cross-stream calibrated disagreement (CSCD).

    Build per-pixel uncertainty from entropy of the mean probability of the two auxiliary views.
    High-uncertainty regions: symmetric KL between auxiliary logits.
    Low-uncertainty regions: L1 agreement of auxiliaries with the main head.
    """
    lr = logits_rgb
    ld = logits_dem
    if lr.shape[-2:] != logits_main.shape[-2:]:
        lr = F.interpolate(lr, logits_main.shape[-2:], mode="bilinear", align_corners=False)
        ld = F.interpolate(ld, logits_main.shape[-2:], mode="bilinear", align_corners=False)

    pr = torch.sigmoid(lr)
    pd = torch.sigmoid(ld)
    pm = 0.5 * (pr + pd)
    ent = -(pm * torch.log(pm + eps) + (1 - pm) * torch.log(1 - pm + eps))
    flat = ent.view(ent.size(0), -1)
    u_thresh = torch.quantile(flat, uncertainty_quantile, dim=1, keepdim=True).view(-1, 1, 1, 1)
    u_mask = (ent >= u_thresh).float()

    skl = symmetric_kl_logits(lr, ld)
    disagree = (skl * u_mask).sum() / (u_mask.sum() + eps)

    pm_main = torch.sigmoid(logits_main)
    agree_r = (pr - pm_main).abs()
    agree_d = (pd - pm_main).abs()
    low = 1.0 - u_mask
    agree = ((agree_r + agree_d) * low).sum() / (low.sum() + eps)

    return disagree + agree_weight * agree
