import torch
import torch.nn as nn
import torch.nn.functional as F

from terrascope.core.image_encoder import Block


class GeoStateBlock(nn.Module):
    """Depthwise terrain propagation on token grids (state-space style inductive bias)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, dim * 2)
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.out_proj = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x_hwc: torch.Tensor) -> torch.Tensor:
        b, h, w, c = x_hwc.shape
        x = self.norm(x_hwc)
        u, g = self.in_proj(x).chunk(2, dim=-1)
        u = u.permute(0, 3, 1, 2)
        u = self.dw_conv(u).permute(0, 2, 3, 1)
        u = u * torch.sigmoid(g)
        return x_hwc + self.gamma * self.out_proj(u)


class JointStreamAttention(nn.Module):
    """
    Multi-head attention where each head's K/V comes from either the RGB or DEM
    stream (split along heads), while Q is stream-specific. Mixes modalities inside MHSA.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        assert num_heads % 2 == 0, "JointStreamAttention requires an even head count"
        self.half_heads = num_heads // 2

        self.rgb_ln = nn.LayerNorm(dim)
        self.dem_ln = nn.LayerNorm(dim)
        self.q_rgb = nn.Linear(dim, dim)
        self.q_dem = nn.Linear(dim, dim)
        self.kv_rgb = nn.Linear(dim, dim * 2)
        self.kv_dem = nn.Linear(dim, dim * 2)
        self.out_rgb = nn.Linear(dim, dim)
        self.out_dem = nn.Linear(dim, dim)

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        return attn @ v

    def forward(
        self, rgb_hwc: torch.Tensor, dem_hwc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, h, w, c = rgb_hwc.shape
        n = h * w
        rgb = rgb_hwc.view(b, n, c)
        dem = dem_hwc.view(b, n, c)
        rn = self.rgb_ln(rgb)
        dn = self.dem_ln(dem)
        gate = torch.sigmoid(self.stream_gate)

        def run_rgb_query():
            q = self.q_rgb(rn).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            kr, vr = self.kv_rgb(rn).chunk(2, dim=-1)
            kd, vd = self.kv_dem(dn).chunk(2, dim=-1)
            kr = kr.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            vr = vr.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            kd = kd.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            vd = vd.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            k = kr * (1.0 - gate) + kd * gate
            v = vr * (1.0 - gate) + vd * gate
            o = self._attn(q, k, v).transpose(1, 2).reshape(b, n, c)
            return self.out_rgb(o)

        def run_dem_query():
            q = self.q_dem(dn).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            kd, vd = self.kv_dem(dn).chunk(2, dim=-1)
            kr, vr = self.kv_rgb(rn).chunk(2, dim=-1)
            kd = kd.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            vd = vd.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            kr = kr.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            vr = vr.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            k = torch.cat([kd[:, :hh], kr[:, hh:]], dim=1)
            v = torch.cat([vd[:, :hh], vr[:, hh:]], dim=1)
            o = self._attn(q, k, v).transpose(1, 2).reshape(b, n, c)
            return self.out_dem(o)

        upd_rgb = run_rgb_query()
        upd_dem = run_dem_query()

        rgb2 = rgb + upd_rgb
        dem2 = dem + upd_dem
        return rgb2.view(b, h, w, c), dem2.view(b, h, w, c)


class CrossStreamFusionBlock(nn.Module):
    """Bi-directional cross-attn plus tri-gated mixing (SAM-era fusion, Terrascope wiring)."""

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
        nh = self.num_heads
        hd = c // nh
        q = q.view(b, n, nh, hd).transpose(1, 2)
        k = k.view(b, n, nh, hd).transpose(1, 2)
        v = v.view(b, n, nh, hd).transpose(1, 2)
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


class CoupledStreamBlock(nn.Module):
    """One Terrascope stage: PEG → joint MHSA → dual SAM-Blocks → fusion → GeoState (RGB)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        norm_layer: type,
        act_layer: type,
        use_rel_pos: bool,
        rel_pos_zero_init: bool,
        window_size: int,
        input_size: tuple[int, int],
        peg_rgb: nn.Module,
        peg_dem: nn.Module,
    ):
        super().__init__()
        self.peg_rgb = peg_rgb
        self.peg_dem = peg_dem
        self.joint_attn = JointStreamAttention(dim, num_heads)
        self.block_rgb = Block(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            norm_layer,
            act_layer,
            use_rel_pos,
            rel_pos_zero_init,
            window_size,
            input_size,
        )
        self.block_dem = Block(
            dim,
            num_heads,
            mlp_ratio,
            qkv_bias,
            norm_layer,
            act_layer,
            use_rel_pos,
            rel_pos_zero_init,
            window_size,
            input_size,
        )
        self.fusion = CrossStreamFusionBlock(dim, num_heads)
        self.geo = GeoStateBlock(dim)
        self.stream_mix_rgb = nn.Linear(dim * 2, dim)
        self.stream_mix_dem = nn.Linear(dim * 2, dim)

    def forward(self, rgb_hwc: torch.Tensor, dem_hwc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, h, w, c = rgb_hwc.shape
        rgb_seq = rgb_hwc.reshape(b, h * w, c)
        dem_seq = dem_hwc.reshape(b, h * w, c)
        rgb_t = self.peg_rgb(rgb_seq, h, w).reshape(b, h, w, c)
        dem_t = self.peg_dem(dem_seq, h, w).reshape(b, h, w, c)

        rgb_t, dem_t = self.joint_attn(rgb_t, dem_t)
        rgb_t = self.block_rgb(rgb_t)
        dem_t = self.block_dem(dem_t)
        rgb_t, dem_t = self.fusion(rgb_t, dem_t)
        fused = self.stream_mix_rgb(torch.cat([rgb_t, dem_t], dim=-1))
        rgb_t = self.geo(fused)
        dem_t = self.stream_mix_dem(torch.cat([dem_t, rgb_t], dim=-1))
        return rgb_t, dem_t
