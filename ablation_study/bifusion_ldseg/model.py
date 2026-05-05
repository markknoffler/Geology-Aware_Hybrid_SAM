from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(ConvBNReLU(in_channels, out_channels), ConvBNReLU(out_channels, out_channels))


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.s1 = DoubleConv(in_channels, c)
        self.s2 = DoubleConv(c, c * 2)
        self.s3 = DoubleConv(c * 2, c * 4)
        self.s4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.channels = [c, c * 2, c * 4, c * 8]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.s1(x)
        f2 = self.s2(self.pool(f1))
        f3 = self.s3(self.pool(f2))
        f4 = self.s4(self.pool(f3))
        return [f1, f2, f3, f4]


class TransformerStage(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        q = self.norm1(tokens)
        attn_out, _ = self.attn(q, q, q)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class TransformerEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.stem = ConvBNReLU(in_channels, c, kernel_size=3)
        self.down1 = ConvBNReLU(c, c * 2, kernel_size=3)
        self.down2 = ConvBNReLU(c * 2, c * 4, kernel_size=3)
        self.down3 = ConvBNReLU(c * 4, c * 8, kernel_size=3)
        self.t1 = TransformerStage(c, num_heads=4)
        self.t2 = TransformerStage(c * 2, num_heads=4)
        self.t3 = TransformerStage(c * 4, num_heads=8)
        self.t4 = TransformerStage(c * 8, num_heads=8)
        self.pool = nn.MaxPool2d(2, 2)
        self.channels = [c, c * 2, c * 4, c * 8]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        f1 = self.t1(self.stem(x))
        f2 = self.t2(self.down1(self.pool(f1)))
        f3 = self.t3(self.down2(self.pool(f2)))
        f4 = self.t4(self.down3(self.pool(f3)))
        return [f1, f2, f3, f4]


class BiAttentionGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.c_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.t_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.c_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.t_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, c_feat: torch.Tensor, t_feat: torch.Tensor):
        c = self.c_proj(c_feat)
        t = self.t_proj(t_feat)
        wc = self.c_gate(torch.cat([c, t], dim=1))
        wt = self.t_gate(torch.cat([t, c], dim=1))
        c_out = c_feat + wc * t
        t_out = t_feat + wt * c
        return c_out, t_out


class LabelEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBNReLU(1, 32),
            nn.MaxPool2d(2, 2),
            ConvBNReLU(32, 64),
            nn.MaxPool2d(2, 2),
            ConvBNReLU(64, latent_dim),
            nn.MaxPool2d(2, 2),
            ConvBNReLU(latent_dim, latent_dim),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        z = self.enc(mask)
        mean = z.mean(dim=(1, 2, 3), keepdim=True)
        std = z.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return (z - mean) / std


class ConditionalDenoiser(nn.Module):
    def __init__(self, latent_dim: int = 128, cond_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.cond_proj = nn.Conv2d(cond_dim, latent_dim, kernel_size=1)
        self.net = nn.Sequential(
            ConvBNReLU(latent_dim * 2, latent_dim),
            ConvBNReLU(latent_dim, latent_dim),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
        )

    def forward(self, zt: torch.Tensor, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        b, c, h, w = zt.shape
        t_embed = self.time_mlp(t[:, None]).view(b, c, 1, 1)
        cond = F.interpolate(cond, size=(h, w), mode="bilinear", align_corners=False)
        cond = self.cond_proj(cond)
        x = torch.cat([zt + t_embed, cond], dim=1)
        return self.net(x)


class ReverseAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = DoubleConv(out_channels + skip_channels + out_channels, out_channels)
        self.pred = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, prev_pred: torch.Tensor):
        x = self.up(x)
        prev_up = F.interpolate(prev_pred, size=x.shape[-2:], mode="bilinear", align_corners=False)
        attention = 1.0 - torch.sigmoid(prev_up)
        ra_feat = x * attention
        out = self.refine(torch.cat([ra_feat, skip, x], dim=1))
        pred = self.pred(out)
        return out, pred


class BiFusionLDSeg(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, latent_dim: int = 128, diffusion_steps: int = 1000):
        super().__init__()
        self.diffusion_steps = diffusion_steps
        self.cnn = CNNEncoder(in_channels=in_channels, base_channels=base_channels)
        self.trans = TransformerEncoder(in_channels=in_channels, base_channels=base_channels)
        ch = self.cnn.channels
        self.biag = nn.ModuleList([BiAttentionGate(c) for c in ch])

        fused_deep_channels = ch[-1] * 2
        self.cond_proj = nn.Conv2d(fused_deep_channels, latent_dim * 2, kernel_size=1)
        self.label_encoder = LabelEncoder(latent_dim=latent_dim)
        self.denoiser = ConditionalDenoiser(latent_dim=latent_dim, cond_dim=latent_dim * 2)
        self.z_to_dec = ConvBNReLU(latent_dim, ch[-1], kernel_size=3)

        self.coarse_pred = nn.Conv2d(ch[-1], 1, kernel_size=1)
        self.dec3 = ReverseAttentionBlock(ch[-1], ch[-2] * 2, ch[-2])
        self.dec2 = ReverseAttentionBlock(ch[-2], ch[-3] * 2, ch[-3])
        self.dec1 = ReverseAttentionBlock(ch[-3], ch[-4] * 2, ch[-4])
        self.final_head = nn.Conv2d(ch[-4], 1, kernel_size=1)

    def _fuse_features(self, x: torch.Tensor):
        c_feats = self.cnn(x)
        t_feats = self.trans(x)
        fused = []
        for i in range(len(c_feats)):
            c, t = self.biag[i](c_feats[i], t_feats[i])
            fused.append(torch.cat([c, t], dim=1))
        return fused

    def _sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randint(1, self.diffusion_steps + 1, (batch_size,), device=device).float() / float(self.diffusion_steps)

    def _q_sample(self, z0: torch.Tensor, t: torch.Tensor):
        alpha_bar = torch.cos((t * 0.5 + 0.5) * torch.pi / 2.0) ** 2
        alpha_bar = alpha_bar.view(-1, 1, 1, 1).clamp(1e-4, 1.0)
        eps = torch.randn_like(z0)
        zt = torch.sqrt(alpha_bar) * z0 + torch.sqrt(1.0 - alpha_bar) * eps
        return zt, eps

    def decode(self, zdn: torch.Tensor, fused_features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        f1, f2, f3, f4 = fused_features
        x4 = self.z_to_dec(zdn)
        p4 = self.coarse_pred(x4)
        x3, p3 = self.dec3(x4, f3, p4)
        x2, p2 = self.dec2(x3, f2, p3)
        x1, p1 = self.dec1(x2, f1, p2)
        p0 = self.final_head(x1)
        return {"main": p0, "aux1": p1, "aux2": p2, "aux3": p3, "aux4": p4}

    def forward(self, image: torch.Tensor, mask: torch.Tensor = None, sample_steps: int = 10):
        fused = self._fuse_features(image)
        cond = self.cond_proj(fused[-1])

        if self.training and mask is not None:
            z0 = self.label_encoder(mask)
            t = self._sample_t(image.shape[0], image.device)
            zt, eps = self._q_sample(z0, t)
            eps_pred = self.denoiser(zt, cond, t)
            zdn = zt - eps_pred
            preds = self.decode(zdn, fused)
            preds["eps_pred"] = eps_pred
            preds["eps_true"] = eps
            return preds

        b = image.shape[0]
        z = torch.randn((b, cond.shape[1] // 2, cond.shape[2], cond.shape[3]), device=image.device)
        for i in range(sample_steps, 0, -1):
            t = torch.full((b,), float(i) / float(sample_steps), device=image.device)
            eps_pred = self.denoiser(z, cond, t)
            z = z - eps_pred / float(sample_steps)
        return self.decode(z, fused)
