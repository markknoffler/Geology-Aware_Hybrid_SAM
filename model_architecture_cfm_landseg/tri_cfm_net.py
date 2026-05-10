from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .decoders import VelocityConditionalUNet
from .encoders import MultiScaleFusionStack, PyramidEncoder
from .losses.composite import compute_fm_residuals_batch, mask_to_latent_z


class TriEncoderCFMNet(nn.Module):
    """Triple-stream encoders → multi-scale gated fusion → aux segmentation + conditional FM."""

    def __init__(
        self,
        rgb_ch: int = 3,
        dem_ch: int = 1,
        ctx_ch: int = 6,
        pyramid_width: int = 48,
        flow_combine_scale: float = 0.5,
        inference_flow_steps: int = 0,
        latent_sigma: float = 4.0,
    ):
        super().__init__()
        self.rgb_enc = PyramidEncoder(rgb_ch, width=pyramid_width, dem_branch=False)
        self.dem_enc = PyramidEncoder(dem_ch, width=pyramid_width, dem_branch=True)
        self.ctx_enc = PyramidEncoder(ctx_ch, width=pyramid_width, dem_branch=False)
        chs = list(self.rgb_enc.out_channels)

        self.fusion = MultiScaleFusionStack(chs)
        fuse_chs = tuple(chs)
        self.velocity = VelocityConditionalUNet(fuse_channels=fuse_chs)
        hidden = chs[0]
        self.aux_head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

        self.flow_readout = nn.Conv2d(1, 1, kernel_size=1, bias=True)
        self.flow_combine_scale = flow_combine_scale
        self.default_flow_steps = inference_flow_steps
        self.latent_sigma = latent_sigma

    def _encode_fused(self, stream_rgb: torch.Tensor, stream_dem: torch.Tensor, stream_ctx: torch.Tensor):
        r = self.rgb_enc(stream_rgb)
        d = self.dem_enc(stream_dem)
        c = self.ctx_enc(stream_ctx)
        fused = self.fusion(r, d, c)
        return fused, r

    def _aux_logits_fullres(self, fused: List[torch.Tensor], spatial_target: tuple[int, int]):
        logits_lr = self.aux_head(fused[0])
        return torch.nn.functional.interpolate(logits_lr, size=spatial_target, mode="bilinear", align_corners=False)

    @torch.no_grad()
    def integrate_flow_logits(self, fused: List[torch.Tensor], shape_hw: tuple[int, int], steps: int) -> torch.Tensor:
        steps = max(int(steps), 1)
        b = fused[0].shape[0]
        h, w = shape_hw
        dtype = fused[0].dtype
        device = fused[0].device
        eps = torch.randn(b, 1, h, w, device=device, dtype=dtype)
        x = eps
        dt = 1.0 / steps
        for k in range(steps):
            t_scalar = max(1.0 - float(k) / float(steps), 1e-3)
            t_vec = eps.new_full((b,), t_scalar)
            x = x - dt * self.velocity(x, t_vec, fused)
        return self.flow_readout(x)

    def forward(
        self,
        stream_rgb: torch.Tensor,
        stream_dem: torch.Tensor,
        stream_ctx: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
        *,
        inference_flow_steps: Optional[int] = None,
    ):
        fused, _ = self._encode_fused(stream_rgb, stream_dem, stream_ctx)
        h_tgt, w_tgt = stream_rgb.shape[-2:]
        logits_aux = self._aux_logits_fullres(fused, (h_tgt, w_tgt))

        out: Dict[str, torch.Tensor] = {"logits_aux": logits_aux}

        fm_residual = logits_aux.new_zeros(())
        smooth_penalty = logits_aux.new_zeros(())

        if self.training and gt_mask is not None:
            latent_z = mask_to_latent_z(gt_mask, sigma=self.latent_sigma)
            if latent_z.shape[-2:] != (h_tgt, w_tgt):
                latent_z = torch.nn.functional.interpolate(latent_z, size=(h_tgt, w_tgt), mode="nearest")
            epsilon = torch.randn_like(latent_z)
            batch = latent_z.shape[0]
            t = torch.rand(batch, device=latent_z.device, dtype=latent_z.dtype)

            xt = tensor_lincomb(latent_z, epsilon, t)
            v_pred = self.velocity(xt, t, fused)
            fm_residual = compute_fm_residuals_batch(v_pred, epsilon, latent_z).mean()

            t2 = torch.clamp(t + 0.1, max=1.0)
            v_second = self.velocity(xt, t2, fused)
            smooth_penalty = torch.mean((v_pred - v_second) ** 2)

            out["fm_residual"] = fm_residual
            out["v_smooth_penalty"] = smooth_penalty

        logits_out = logits_aux
        if not self.training:
            infer_steps = (
                inference_flow_steps if inference_flow_steps is not None else int(self.default_flow_steps)
            )
            if infer_steps > 0:
                lf = self.integrate_flow_logits(fused, (h_tgt, w_tgt), infer_steps)
                logits_out = logits_aux + self.flow_combine_scale * lf

        out["logits_aux"] = logits_out
        return out


def tensor_lincomb(z: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    tb = t.view(-1, 1, 1, 1)
    return (1.0 - tb) * z + tb * eps
