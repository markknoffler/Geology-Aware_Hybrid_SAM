from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 1, base_ch: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        self.up4 = UpBlock(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up3 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = UpBlock(base_ch * 2, base_ch, base_ch)
        self.head = nn.Conv2d(base_ch, n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)


class DualStreamUNetBaseline(nn.Module):
    def __init__(self, n_classes: int = 1):
        super().__init__()
        self.s1 = UNetBaseline(in_channels=3, n_classes=16, base_ch=16)
        self.s2 = UNetBaseline(in_channels=3, n_classes=16, base_ch=16)
        self.out = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x1, x2):
        f1 = self.s1(x1)
        f2 = self.s2(x2)
        return self.out(torch.cat([f1, f2], dim=1))


class LinkNetBaseline(nn.Module):
    """
    Lightweight LinkNet-style encoder-decoder.
    """

    def __init__(self, in_channels: int = 3, n_classes: int = 1, base_ch: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, base_ch, 3, stride=2, padding=1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv2d(base_ch * 4, base_ch * 8, 3, stride=2, padding=1), nn.ReLU(True))
        self.dec3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec0 = nn.ConvTranspose2d(base_ch, base_ch, 2, stride=2)
        self.head = nn.Conv2d(base_ch, n_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d3 = F.relu(self.dec3(e4) + e3)
        d2 = F.relu(self.dec2(d3) + e2)
        d1 = F.relu(self.dec1(d2) + e1)
        d0 = F.relu(self.dec0(d1))
        return self.head(d0)


class DeepLabV3PlusBaseline(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 1, pretrained: bool = False):
        super().__init__()
        weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights, num_classes=n_classes)
        if in_channels != 3:
            self.model.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

    def forward(self, x):
        return self.model(x)["out"]


class AttentionBottleneck(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=4, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(channels, channels * 2), nn.GELU(), nn.Linear(channels * 2, channels))

    def forward(self, x):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t2, _ = self.attn(self.norm(t), self.norm(t), self.norm(t))
        t = t + t2
        t = t + self.mlp(self.norm(t))
        return t.transpose(1, 2).reshape(b, c, h, w)


class TransformerUNetBaseline(UNetBaseline):
    def __init__(self, in_channels: int = 3, n_classes: int = 1, base_ch: int = 32):
        super().__init__(in_channels=in_channels, n_classes=n_classes, base_ch=base_ch)
        self.bottleneck_attn = AttentionBottleneck(base_ch * 16)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        b = self.bottleneck_attn(b)
        d4 = self.up4(b, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return self.head(d1)


def build_model(model_name: str, in_channels: int = 3, n_classes: int = 1):
    name = model_name.lower()
    if name == "unet":
        return UNetBaseline(in_channels=in_channels, n_classes=n_classes, base_ch=32)
    if name == "dual_stream_unet":
        return DualStreamUNetBaseline(n_classes=n_classes)
    if name == "linknet":
        return LinkNetBaseline(in_channels=in_channels, n_classes=n_classes, base_ch=32)
    if name == "deeplabv3plus":
        return DeepLabV3PlusBaseline(in_channels=in_channels, n_classes=n_classes, pretrained=False)
    if name in {"transunet", "shapeformer", "rmau_net", "dep_unet", "emr_hrnet", "gmnet"}:
        # Approximate transformer/attention baselines with a shared transformer-UNet core.
        return TransformerUNetBaseline(in_channels=in_channels, n_classes=n_classes, base_ch=32)
    raise ValueError(f"Unsupported baseline model: {model_name}")
