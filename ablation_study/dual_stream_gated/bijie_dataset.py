from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


EPSILON = 1e-6


def _load_png(fp: Path) -> np.ndarray:
    # Keep bit-depth (PIL preserves uint8/uint16 depending on source)
    arr = np.array(Image.open(fp))
    return arr


def _normalize_to_01_per_channel(x: torch.Tensor, eps: float = EPSILON) -> torch.Tensor:
    # x: (C,H,W), float tensor
    out = x.clone()
    for c in range(out.shape[0]):
        channel = out[c]
        mn = float(channel.min())
        mx = float(channel.max())
        if mx > mn:
            channel = (channel - mn) / (mx - mn + eps)
        out[c] = torch.clamp(channel, 0.0, 1.0)
    return out


def _resize_chw(img_chw: np.ndarray, size_xy: Tuple[int, int], is_mask: bool) -> np.ndarray:
    # img_chw: (C,H,W)
    if img_chw.ndim != 3:
        raise ValueError(f"Expected CHW, got shape {img_chw.shape}")
    img_hwc = np.transpose(img_chw, (1, 2, 0))  # (H,W,C)
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img_hwc, size_xy, interpolation=interp)
    # OpenCV may drop the channel dimension for single-channel inputs.
    if resized.ndim == 2:
        resized = resized[:, :, None]
    return np.transpose(resized, (2, 0, 1))


class DualStreamTransformBijie:
    """
    Paper-style augmentation for Bijie:
    - random hflip / vflip
    - random Gaussian noise
    - random CLAHE
    No salt-and-pepper (paper doesn't mention it).
    """

    def __init__(self, p: float = 0.5, gaussian_std: float = 0.05, clahe_clip: float = 2.0):
        self.p = p
        self.gaussian_std = gaussian_std
        self.clahe_clip = clahe_clip

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, label: torch.Tensor):
        if random.random() < self.p:
            image1 = TF.hflip(image1)
            image2 = TF.hflip(image2)
            label = TF.hflip(label)
        if random.random() < self.p:
            image1 = TF.vflip(image1)
            image2 = TF.vflip(image2)
            label = TF.vflip(label)

        if random.random() < self.p:
            noise1 = torch.randn_like(image1) * self.gaussian_std
            noise2 = torch.randn_like(image2) * self.gaussian_std
            image1 = image1 + noise1
            image2 = image2 + noise2

        if random.random() < self.p:
            image1 = self._apply_clahe(image1)
            image2 = self._apply_clahe(image2)

        return image1, image2, label

    def _apply_clahe(self, tensor_img: torch.Tensor) -> torch.Tensor:
        # tensor_img expected in [0,1] float
        img = tensor_img.detach().cpu().numpy()
        img_hwc = np.transpose(img, (1, 2, 0))
        img_u8 = np.clip(img_hwc * 255.0, 0, 255).astype(np.uint8)

        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=(8, 8))
        for c in range(img_u8.shape[2]):
            img_u8[:, :, c] = clahe.apply(img_u8[:, :, c])

        img_f = img_u8.astype(np.float32) / 255.0
        out = np.transpose(img_f, (2, 0, 1))
        return torch.from_numpy(out).type_as(tensor_img)


class BijieRawDataset(Dataset):
    """
    Loads (image, dem, mask) from Bijie PNG structure.

    Expected directory layout:
      root/
        image/*.png
        dem/*.png
        mask/*.png  (only for landslide phase)
    """

    def __init__(self, root_dir: str | Path, phase: str):
        phase = phase.lower()
        if phase not in {"landslide", "non-landslide"}:
            raise ValueError(f"Unknown phase: {phase}")

        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / "image"
        self.dem_dir = self.root_dir / "dem"
        self.mask_dir = None if phase == "non-landslide" else (self.root_dir / "mask")

        self.files = sorted(f for f in self.image_dir.iterdir() if f.suffix.lower() == ".png")
        if not self.files:
            raise FileNotFoundError(f"No PNGs found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        img_fp = self.files[idx]
        name = img_fp.stem
        dem_fp = self.dem_dir / f"{name}.png"

        image = _load_png(img_fp)  # likely (H,W,3)
        dem = _load_png(dem_fp)  # likely (H,W) or (H,W,1)

        if self.mask_dir is None:
            # no mask for non-landslide
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask_fp = self.mask_dir / f"{name}.png"
            mask = _load_png(mask_fp)

        # Ensure shapes are usable
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.ndim == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        if dem.ndim == 3:
            dem = dem[:, :, 0]

        if mask.ndim == 3:
            mask = mask[:, :, 0]

        return {"image": image, "dem": dem, "mask": mask}


class BijieTwoComposites(Dataset):
    """
    Wraps BijieRawDataset and produces:
      - stream_a: RGB (3,H,W)
      - stream_b: DEM replicated to 3 channels (3,H,W)
      - mask: (1,H,W) float in {0,1}
    """

    def __init__(
        self,
        dataset: Dataset,
        resize_to: int = 256,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.resize_to = resize_to
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.dataset[idx]
        image = sample["image"]  # (H,W,3)
        dem = sample["dem"]  # (H,W)
        mask = sample["mask"]  # (H,W)

        # Resize image
        image_chw = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image_chw = _resize_chw(image_chw, (self.resize_to, self.resize_to), is_mask=False).astype(np.float32)

        # Resize DEM
        dem_chw = np.expand_dims(dem.astype(np.float32), axis=0)  # (1,H,W)
        dem_chw = _resize_chw(dem_chw, (self.resize_to, self.resize_to), is_mask=False).astype(np.float32)

        # Resize mask (nearest)
        mask_chw = np.expand_dims(mask.astype(np.float32), axis=0)  # (1,H,W)
        mask_chw = _resize_chw(mask_chw, (self.resize_to, self.resize_to), is_mask=True)

        # Stream A: RGB
        stream_a = torch.from_numpy(image_chw).float()

        # Stream B: DEM replicated 3x to match model input stream-B
        stream_b_single = torch.from_numpy(dem_chw).float()  # (1,H,W)
        stream_b = stream_b_single.repeat(3, 1, 1)  # (3,H,W)

        # Mask: binary
        mask_tensor = torch.from_numpy(mask_chw).float()
        mask_tensor = (mask_tensor > 0).float()

        # Normalize each channel to [0,1] like official repo
        stream_a = _normalize_to_01_per_channel(stream_a)
        stream_b = _normalize_to_01_per_channel(stream_b)

        if self.transform is not None:
            stream_a, stream_b, mask_tensor = self.transform(stream_a, stream_b, mask_tensor)

        return {"stream_a": stream_a, "stream_b": stream_b, "mask": mask_tensor}

