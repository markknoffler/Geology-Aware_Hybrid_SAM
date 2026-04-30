# cnn_adapter.py ────────────────────────────────────────────────────────────
# plug-and-play CNN branch with **geology-aware recalibration**  ❱❱ feeds GSAM

import torch
import torch.nn as nn
import torchvision.models as tv


# -------------------------------------------------------------------------- #
# 1.  BACKBONE CATALOG – choose the trunk that extracts the *raw* features   #
# -------------------------------------------------------------------------- #
_BACKBONES = {
    # name           → (torchvision-factory, channels @ /16, native stride)
    "convnext_base":   (tv.convnext_base,       1024, 16),
    "efficientnet_b4": (tv.efficientnet_b4,     1792, 16),
    "darknet53":       (lambda w=None:
                        tv.models.darknet_darknet53(weights=None), 1024, 16),
}


# -------------------------------------------------------------------------- #
# 2.  GEOLOGY–AWARE ATTENTION block                                          #
#     (lightweight, self-supervised – no external terrain labels required)   #
# -------------------------------------------------------------------------- #
class _GeoAttention(nn.Module):
    """
    Learns a per-pixel, per-channel gate that highlights textures typical of
    the underlying terrain.  A single, *scalar* α decides how strongly the
    geology map contributes; α is optimised together with all other weights.
    """
    def __init__(self, channels: int):
        super().__init__()
        # depth-wise 3×3 → squeeze-->σ  →  soft mask 0-1
        self.dw_mask   = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )
        # learnable global scale (starts at 0 → network can ignore geology
        # until the signal is useful)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        mask = self.dw_mask(feat)                  # B×C×H×W  in  0-1 range
        geo  = feat * mask                         # gated geology map
        return feat + self.alpha * geo             # keep *all* original info
                                                   # add geology-aware boost


# -------------------------------------------------------------------------- #
# 3.  MAIN ADAPTER – backbone → 1×1 proj → geology gate                      #
# -------------------------------------------------------------------------- #
class CNNAdapter(nn.Module):
    """
    Produces a 1/16-scale  feature grid  (B × embed_dim × H/16 × W/16)
    enriched with geology-aware attention.  The output keeps **exactly** the
    same shape SAM expects, so the rest of the codebase stays untouched.
    """
    def __init__(
        self,
        embed_dim: int,              # 768 for ViT-B
        backbone:  str = "convnext_base",
        pretrained: bool = True,
    ):
        super().__init__()

        if backbone not in _BACKBONES:
            raise ValueError(f"unknown backbone '{backbone}'")

        factory, out_c, stride = _BACKBONES[backbone]
        if stride != 16:
            raise ValueError("adapter output must be at stride 16 (H/16 × W/16)")

        # ------------ build truncated backbone (kept frozen/unfrozen by builder)
        cnn_full = factory(weights="DEFAULT" if pretrained else None)

        if "convnext" in backbone:
            self.stem = cnn_full.features[:7]                       # /16
        elif "efficientnet" in backbone:
            self.stem = nn.Sequential(                              # /2  /4  /8  /16
                cnn_full.features[0],
                cnn_full.features[1],
                cnn_full.features[2],
                cnn_full.features[3],
            )
        elif "darknet" in backbone:
            self.stem = nn.Sequential(                              # /2  /4  /8  /16
                cnn_full.layers[:2],
                cnn_full.layers[2],
                cnn_full.layers[3],
            )
        else:
            raise NotImplementedError

        # 1×1 projection → match ViT embed dimension
        self.proj = nn.Conv2d(out_c, embed_dim, 1)

        # geology-aware attention gate
        self.geo_gate = _GeoAttention(embed_dim)

    # ------------------------------------------------------------------ #
    # forward                                                            #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        torch.Tensor
            B × embed_dim × H/16 × W/16   (H/16 = W/16 = 32 for 512² inputs)
        """
        feat = self.stem(x)        # B×C_out×H/16×W/16
        feat = self.proj(feat)     # B×embed_dim×H/16×W/16

        # add geology-aware boost (controlled by learnable scalar α)
        feat = self.geo_gate(feat)

        return feat
