from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import h5py
import numpy as np
import random
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

EPSILON = 1e-6


def _minmax_per_channel(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    for c in range(out.shape[0]):
        mn = float(out[c].min())
        mx = float(out[c].max())
        if mx > mn:
            out[c] = (out[c] - mn) / (mx - mn + EPSILON)
        out[c] = torch.clamp(out[c], 0.0, 1.0)
    return out


class AugmentTripleStream:
    def __init__(self, p: float = 0.5):
        self.p = p

    @staticmethod
    def _clahe(x: torch.Tensor) -> torch.Tensor:
        arr = x.detach().cpu().numpy()
        arr = np.transpose(arr, (1, 2, 0))
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for c in range(arr.shape[2]):
            arr[:, :, c] = clahe.apply(arr[:, :, c])
        arr = arr.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr).type_as(x)

    def __call__(
        self, rgb: torch.Tensor, dem: torch.Tensor, ctx: torch.Tensor, y: torch.Tensor
    ):
        if random.random() < self.p:
            rgb = TF.hflip(rgb)
            dem = TF.hflip(dem)
            ctx = TF.hflip(ctx)
            y = TF.hflip(y)
        if random.random() < self.p:
            rgb = TF.vflip(rgb)
            dem = TF.vflip(dem)
            ctx = TF.vflip(ctx)
            y = TF.vflip(y)
        if random.random() < self.p:
            rgb = rgb + torch.randn_like(rgb) * 0.05
            ctx = ctx + torch.randn_like(ctx) * 0.05
        if random.random() < self.p:
            rgb = self._clahe(rgb)
            ctx = self._clahe(ctx)
        return rgb, dem, ctx, y


class L4STripleStreamDataset(Dataset):
    """
    Landslide4Sense triple stream:
      stream_rgb = RGB (B4,B3,B2)
      stream_dem = DEM channel only (B14)
      stream_ctx  = concat(RGB, NDVI, slope, DEM) = 6 channels
    """

    def __init__(
        self,
        dataset_root: str | Path,
        ids: Sequence[int],
        resize_to: int = 256,
        transform: Optional[AugmentTripleStream] = None,
    ):
        self.root = Path(dataset_root)
        self.resize_to = resize_to
        self.transform = transform

        img_dir = self.root / "TrainData" / "img"
        mask_dir = self.root / "TrainData" / "mask"
        files = sorted(img_dir.glob("*.h5"))
        self.img_paths = [files[i] for i in ids]
        self.mask_paths = [mask_dir / p.name.replace("image_", "mask_") for p in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    @staticmethod
    def _read_h5(path: Path, key_hint: str):
        with h5py.File(path, "r") as f:
            if key_hint in f:
                return np.asarray(f[key_hint], dtype=np.float32)
            for k in f.keys():
                arr = np.asarray(f[k], dtype=np.float32)
                if arr.ndim >= 2:
                    return arr
        raise ValueError(f"No array found in {path}")

    def __getitem__(self, idx: int):
        image = self._read_h5(self.img_paths[idx], "img")
        mask = self._read_h5(self.mask_paths[idx], "mask")
        if image.ndim == 3 and image.shape[0] > 20:
            image = image.transpose(2, 0, 1)
        elif image.ndim == 3 and image.shape[-1] <= 20:
            image = image.transpose(2, 0, 1)
        if mask.ndim == 3:
            mask = mask[0] if mask.shape[0] == 1 else mask[..., 0]

        B2, B3, B4 = image[1], image[2], image[3]
        B8 = image[7]
        B13, B14 = image[12], image[13]
        ndvi = np.clip((B8 - B4) / (B8 + B4 + EPSILON), -1.0, 1.0)
        xa = np.stack([B4, B3, B2], axis=0).astype(np.float32)
        xb = np.stack([ndvi, B13, B14], axis=0).astype(np.float32)
        y = (mask > 0).astype(np.float32)

        xa_hwc = np.transpose(xa, (1, 2, 0))
        xb_hwc = np.transpose(xb, (1, 2, 0))
        xa = np.transpose(cv2.resize(xa_hwc, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR), (2, 0, 1))
        xb = np.transpose(cv2.resize(xb_hwc, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR), (2, 0, 1))
        y = cv2.resize(y, (self.resize_to, self.resize_to), interpolation=cv2.INTER_NEAREST)

        rgb = _minmax_per_channel(torch.from_numpy(xa).float())
        xb_t = _minmax_per_channel(torch.from_numpy(xb).float())
        dem = xb_t[2:3]
        ctx = torch.cat([rgb, xb_t], dim=0)
        y = torch.from_numpy(y[None, ...]).float()

        if self.transform is not None:
            rgb, dem, ctx, y = self.transform(rgb, dem, ctx, y)
        return {"stream_rgb": rgb, "stream_dem": dem, "stream_ctx": ctx, "mask": y}


class BijieTripleStreamDataset(Dataset):
    """Bijie: RGB, single-channel DEM, and ctx = RGB||DEM (4 ch)."""

    def __init__(self, base_dataset: Dataset, resize_to: int = 256, transform: Optional[AugmentTripleStream] = None):
        self.ds = base_dataset
        self.resize_to = resize_to
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        sample = self.ds[idx]
        img = sample["image"]
        dem = sample["dem"]
        mask = sample["mask"]

        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        if dem.ndim == 3:
            dem = dem[:, :, 0]
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        rgb_hw = np.transpose(img, (2, 0, 1)).astype(np.float32)
        dem_a = dem.astype(np.float32)[None, ...]
        y = (mask > 0).astype(np.float32)

        rgb_hwc = np.transpose(rgb_hw, (1, 2, 0))
        dem_hwc = np.transpose(dem_a, (1, 2, 0))
        rgb_hwc = cv2.resize(rgb_hwc, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR)
        dem_hwc = cv2.resize(dem_hwc, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR)
        # OpenCV may squeeze single-channel arrays to HxW after resize.
        if dem_hwc.ndim == 2:
            dem_hwc = dem_hwc[:, :, None]
        y = cv2.resize(y, (self.resize_to, self.resize_to), interpolation=cv2.INTER_NEAREST)

        rgb = _minmax_per_channel(torch.from_numpy(np.transpose(rgb_hwc, (2, 0, 1))).float())
        dem_t = _minmax_per_channel(torch.from_numpy(np.transpose(dem_hwc, (2, 0, 1))).float())
        ctx = torch.cat([rgb, dem_t], dim=0)
        mask_t = torch.from_numpy(y[None, ...]).float()

        if self.transform is not None:
            rgb, dem_t, ctx, mask_t = self.transform(rgb, dem_t, ctx, mask_t)
        return {"stream_rgb": rgb, "stream_dem": dem_t, "stream_ctx": ctx, "mask": mask_t}
