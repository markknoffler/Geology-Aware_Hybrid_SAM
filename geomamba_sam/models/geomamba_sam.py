import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.build_sam import build_sam_vit_b
from segment_anything.modeling.image_encoder import ImageEncoderViT
from peg import PEG


class GeoStateBlock(nn.Module):
    """State-space style terrain propagation on token grids."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x_hwc: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x_hwc.shape
        x = self.norm(x_hwc)
        u, g = self.in_proj(x).chunk(2, dim=-1)
        u = u.permute(0, 3, 1, 2)
        u = self.dw_conv(u).permute(0, 2, 3, 1)
        u = u * torch.sigmoid(g)
        return x_hwc + self.gamma * self.out_proj(u)


class CrossModalFusionBlock(nn.Module):
    """Token-level RGB/DEM cross attention with multi-gate modulation."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.rgb_norm = nn.LayerNorm(dim)
        self.dem_norm = nn.LayerNorm(dim)
        self.rgb_q = nn.Linear(dim, dim)
        self.rgb_kv = nn.Linear(dim, dim * 2)
        self.dem_q = nn.Linear(dim, dim)
        self.dem_kv = nn.Linear(dim, dim * 2)

        self.rgb_out = nn.Linear(dim, dim)
        self.dem_out = nn.Linear(dim, dim)

        self.channel_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.spatial_gate_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.conf_proj = nn.Linear(1, dim)
        self.mix = nn.Linear(dim * 2, dim)

    def _xattn(self, q, k, v):
        b, n, c = q.shape
        q = q.view(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        k = k.view(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        v = v.view(b, n, self.num_heads, c // self.num_heads).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = attn @ v
        return out.transpose(1, 2).contiguous().view(b, n, c)

    def forward(self, rgb_hwc: torch.Tensor, dem_hwc: torch.Tensor):
        b, h, w, c = rgb_hwc.shape
        n = h * w
        rgb = rgb_hwc.view(b, n, c)
        dem = dem_hwc.view(b, n, c)

        rgb_n = self.rgb_norm(rgb)
        dem_n = self.dem_norm(dem)

        rgb_q = self.rgb_q(rgb_n)
        rgb_k, rgb_v = self.dem_kv(dem_n).chunk(2, dim=-1)
        dem_q = self.dem_q(dem_n)
        dem_k, dem_v = self.rgb_kv(rgb_n).chunk(2, dim=-1)

        rgb_ctx = self.rgb_out(self._xattn(rgb_q, rgb_k, rgb_v))
        dem_ctx = self.dem_out(self._xattn(dem_q, dem_k, dem_v))

        rgb_pool = rgb_n.mean(dim=1)
        dem_pool = dem_n.mean(dim=1)
        c_gate = self.channel_gate(torch.cat([rgb_pool, dem_pool], dim=-1)).unsqueeze(1)

        rgb_2d = rgb_n.view(b, h, w, c).permute(0, 3, 1, 2)
        dem_2d = dem_n.view(b, h, w, c).permute(0, 3, 1, 2)
        spatial_feat = torch.cat([rgb_2d.mean(dim=1, keepdim=True), dem_2d.mean(dim=1, keepdim=True)], dim=1)
        s_gate = torch.sigmoid(self.spatial_gate_conv(spatial_feat)).flatten(2).transpose(1, 2)

        dem_std = dem_2d.std(dim=1, keepdim=True).flatten(2).transpose(1, 2)
        conf_gate = torch.sigmoid(self.conf_proj(dem_std))

        gate = c_gate * s_gate * conf_gate
        fused_rgb = rgb + gate * rgb_ctx
        fused_dem = dem + (1.0 - gate) * dem_ctx

        rgb_final = self.mix(torch.cat([fused_rgb, fused_dem], dim=-1)).view(b, h, w, c)
        dem_final = fused_dem.view(b, h, w, c)
        return rgb_final, dem_final


class DEMPatchEmbed(nn.Module):
    def __init__(self, patch_embed: nn.Module):
        super().__init__()
        proj = patch_embed.proj
        self.proj = nn.Conv2d(1, proj.out_channels, kernel_size=proj.kernel_size, stride=proj.stride, padding=proj.padding)
        self.norm = getattr(patch_embed, "norm", None)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class GeoMambaSAMEncoder(ImageEncoderViT):
    """SAM image encoder reworked to natively model RGB+DEM streams."""

    def __init__(self, *args, **kwargs):
        kwargs["use_abs_pos"] = False
        super().__init__(*args, **kwargs)
        self.embed_dim = self.patch_embed.proj.out_channels
        self.dem_patch_embed = DEMPatchEmbed(self.patch_embed)

        heads = self.blocks[0].attn.num_heads
        self.rgb_pegs = nn.ModuleList([PEG(self.embed_dim) for _ in self.blocks])
        self.dem_pegs = nn.ModuleList([PEG(self.embed_dim) for _ in self.blocks])
        self.fusion_blocks = nn.ModuleList([CrossModalFusionBlock(self.embed_dim, heads) for _ in self.blocks])
        self.state_blocks = nn.ModuleList([GeoStateBlock(self.embed_dim) for _ in self.blocks])

        self.dem_to_rgb = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, rgb: torch.Tensor, dem: torch.Tensor) -> torch.Tensor:
        rgb_t = self.patch_embed(rgb)
        dem_t = self.dem_patch_embed(dem)
        b, h, w, c = rgb_t.shape

        for i, blk in enumerate(self.blocks):
            rgb_seq = rgb_t.reshape(b, h * w, c)
            dem_seq = dem_t.reshape(b, h * w, c)
            rgb_t = self.rgb_pegs[i](rgb_seq, h, w).reshape(b, h, w, c)
            dem_t = self.dem_pegs[i](dem_seq, h, w).reshape(b, h, w, c)
            rgb_t, dem_t = self.fusion_blocks[i](rgb_t, dem_t)
            rgb_t = self.state_blocks[i](rgb_t)
            rgb_t = blk(rgb_t)

        rgb_t = rgb_t + self.dem_to_rgb(dem_t)
        return self.neck(rgb_t.permute(0, 3, 1, 2))

    @classmethod
    def from_pretrained(cls, old_enc, **extra_cfg):
        embed_dim = old_enc.patch_embed.proj.out_channels
        hidden_dim = old_enc.blocks[0].mlp.lin1.out_features
        mlp_ratio = hidden_dim // embed_dim
        global_ids = [i for i, b in enumerate(old_enc.blocks) if b.window_size == 0]

        genc = cls(
            img_size=old_enc.img_size,
            patch_size=old_enc.patch_embed.proj.kernel_size[0],
            in_chans=3,
            embed_dim=embed_dim,
            depth=len(old_enc.blocks),
            num_heads=old_enc.blocks[0].attn.num_heads,
            mlp_ratio=mlp_ratio,
            use_rel_pos=True,
            window_size=14,
            global_attn_indexes=tuple(global_ids),
            **extra_cfg,
        )

        src = old_enc.state_dict()
        dst = genc.state_dict()
        for k, v in list(src.items()):
            if k in dst and v.shape != dst[k].shape:
                del src[k]
        genc.load_state_dict(src, strict=False)
        return genc


def build_geomamba_sam_vit_b(
    ckpt_path: str,
    freeze_first_k: int = 4,
    train_decoder: bool = True,
):
    sam = build_sam_vit_b(checkpoint=ckpt_path)
    sam.train()
    sam.image_encoder = GeoMambaSAMEncoder.from_pretrained(sam.image_encoder)

    for p in sam.prompt_encoder.parameters():
        p.requires_grad = False

    for i, blk in enumerate(sam.image_encoder.blocks):
        for p in blk.parameters():
            p.requires_grad = i >= freeze_first_k

    if not train_decoder:
        for p in sam.mask_decoder.parameters():
            p.requires_grad = False

    return sam
