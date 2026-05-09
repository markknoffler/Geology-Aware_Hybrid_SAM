from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn

from terrascope.core.blocks import CoupledStreamBlock
from terrascope.core.positional import PEG
from terrascope.core.common import LayerNorm2d
from terrascope.core.image_encoder import PatchEmbed


class DEMPatchEmbed(nn.Module):
    def __init__(self, patch_embed: PatchEmbed, dem_in_chans: int = 1):
        super().__init__()
        proj = patch_embed.proj
        self.proj = nn.Conv2d(
            dem_in_chans,
            proj.out_channels,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            padding=proj.padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).permute(0, 2, 3, 1)
        return x


class TerrascopeEncoder(nn.Module):
    """
    Dual-stream encoder: stacked CoupledStreamBlocks (PEG + joint MHSA + dual ViT blocks
    + fusion + GeoState), lateral skip injection, SAM-style neck.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_rel_pos: bool = True,
        rel_pos_zero_init: bool = True,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),
        skip_tap_layers: Tuple[int, ...] = (3, 7, 11),
        mid_aux_layer: int = 5,
        dem_in_chans: int = 1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        self.mid_aux_layer = mid_aux_layer
        self.skip_tap_layers = skip_tap_layers

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.dem_patch_embed = DEMPatchEmbed(self.patch_embed, dem_in_chans=dem_in_chans)

        token_hw = img_size // patch_size
        input_size = (token_hw, token_hw)

        self.layers = nn.ModuleList()
        for i in range(depth):
            ws = window_size if i not in global_attn_indexes else 0
            self.layers.append(
                CoupledStreamBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=partial(norm_layer, eps=1e-6),
                    act_layer=act_layer,
                    use_rel_pos=use_rel_pos,
                    rel_pos_zero_init=rel_pos_zero_init,
                    window_size=ws,
                    input_size=input_size,
                    peg_rgb=PEG(embed_dim),
                    peg_dem=PEG(embed_dim),
                )
            )

        self.lateral_fuse = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False), nn.GELU()) for _ in skip_tap_layers]
        )
        self.skip_balance = nn.Parameter(torch.zeros(len(skip_tap_layers)))

        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
        )

    def forward(
        self, rgb: torch.Tensor, dem: torch.Tensor, return_mid: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        rgb_t = self.patch_embed(rgb)
        dem_t = self.dem_patch_embed(dem)
        b, h, w, c = rgb_t.shape

        mid_pair: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        skip_feats = []

        for i, layer in enumerate(self.layers):
            rgb_t, dem_t = layer(rgb_t, dem_t)
            if return_mid and i == self.mid_aux_layer:
                mid_pair = (rgb_t.clone(), dem_t.clone())
            if i in self.skip_tap_layers:
                j = self.skip_tap_layers.index(i)
                xi = rgb_t.permute(0, 3, 1, 2)
                skip_feats.append(self.lateral_fuse[j](xi).permute(0, 2, 3, 1))

        for j, s in enumerate(skip_feats):
            rgb_t = rgb_t + torch.sigmoid(self.skip_balance[j]) * s

        out = self.neck(rgb_t.permute(0, 3, 1, 2))
        return out, mid_pair
