from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import random

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split


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


class Augment2D:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: torch.Tensor, y: torch.Tensor):
        if random.random() < self.p:
            x = TF.hflip(x)
            y = TF.hflip(y)
        if random.random() < self.p:
            x = TF.vflip(x)
            y = TF.vflip(y)
        if random.random() < self.p:
            x = x + torch.randn_like(x) * 0.05
        if random.random() < self.p:
            x = self._clahe(x)
        return x, y

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


class AugmentDual2D:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor):
        if random.random() < self.p:
            x1 = TF.hflip(x1)
            x2 = TF.hflip(x2)
            y = TF.hflip(y)
        if random.random() < self.p:
            x1 = TF.vflip(x1)
            x2 = TF.vflip(x2)
            y = TF.vflip(y)
        if random.random() < self.p:
            x1 = x1 + torch.randn_like(x1) * 0.05
            x2 = x2 + torch.randn_like(x2) * 0.05
        if random.random() < self.p:
            x1 = Augment2D._clahe(x1)
            x2 = Augment2D._clahe(x2)
        return x1, x2, y


class L4SBinaryDataset(Dataset):
    """
    Landslide4Sense single-stream dataset from .h5.
    Uses TrainData/img + TrainData/mask only (paper-style holdout for validation).
    """

    def __init__(
        self,
        dataset_root: str | Path,
        ids: Sequence[int],
        resize_to: int = 256,
        input_mode: str = "rgb",
        transform: Optional[Augment2D] = None,
    ):
        self.root = Path(dataset_root)
        self.resize_to = resize_to
        self.transform = transform
        self.input_mode = input_mode.lower()

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

    def _select_channels(self, image_chw: np.ndarray) -> np.ndarray:
        # default mapping from paper repo
        B2, B3, B4 = image_chw[1], image_chw[2], image_chw[3]
        B8, B8A = image_chw[7], image_chw[8]
        B11, B12 = image_chw[10], image_chw[11]
        B13, B14 = image_chw[12], image_chw[13]

        if self.input_mode == "rgb":
            return np.stack([B4, B3, B2], axis=0)
        if self.input_mode == "rgb_swir":
            return np.stack([B4, B3, B2, B11, B12, B8A], axis=0)
        if self.input_mode == "ngb":
            return np.stack([B8, B3, B2], axis=0)
        if self.input_mode == "all14":
            return image_chw
        if self.input_mode == "rgb_ndvi_slope_dem":
            ndvi = np.clip((B8 - B4) / (B8 + B4 + EPSILON), -1.0, 1.0)
            return np.stack([B4, B3, B2, ndvi, B13, B14], axis=0)
        raise ValueError(f"Unknown input_mode: {self.input_mode}")

    def __getitem__(self, idx: int):
        image = self._read_h5(self.img_paths[idx], "img")
        mask = self._read_h5(self.mask_paths[idx], "mask")

        if image.ndim == 3 and image.shape[0] > 20:
            image = image.transpose(2, 0, 1)
        elif image.ndim == 3 and image.shape[-1] <= 20:
            image = image.transpose(2, 0, 1)

        if mask.ndim == 3:
            mask = mask[0] if mask.shape[0] == 1 else mask[..., 0]

        x = self._select_channels(image).astype(np.float32)
        y = (mask > 0).astype(np.float32)

        x_hwc = np.transpose(x, (1, 2, 0))
        x_hwc = cv2.resize(x_hwc, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR)
        x = np.transpose(x_hwc, (2, 0, 1))
        y = cv2.resize(y, (self.resize_to, self.resize_to), interpolation=cv2.INTER_NEAREST)

        x = _minmax_per_channel(torch.from_numpy(x).float())
        y = torch.from_numpy(y[None, ...]).float()
        if self.transform is not None:
            x, y = self.transform(x, y)
        return {"image": x, "mask": y}


class L4SDualStreamDataset(Dataset):
    """
    Landslide4Sense dual-stream:
      stream_a = RGB
      stream_b = NDVI + slope + DEM
    """

    def __init__(
        self,
        dataset_root: str | Path,
        ids: Sequence[int],
        resize_to: int = 256,
        transform: Optional[AugmentDual2D] = None,
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

        xa = _minmax_per_channel(torch.from_numpy(xa).float())
        xb = _minmax_per_channel(torch.from_numpy(xb).float())
        y = torch.from_numpy(y[None, ...]).float()

        if self.transform is not None:
            xa, xb, y = self.transform(xa, xb, y)
        return {"stream_a": xa, "stream_b": xb, "mask": y}


class BijieSingleStreamDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        resize_to: int = 256,
        input_mode: str = "rgb",
        transform: Optional[Augment2D] = None,
    ):
        self.ds = base_dataset
        self.resize_to = resize_to
        self.input_mode = input_mode.lower()
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

        if self.input_mode == "rgb":
            x = np.transpose(img, (2, 0, 1)).astype(np.float32)
        elif self.input_mode == "rgb_dem":
            x_rgb = np.transpose(img, (2, 0, 1)).astype(np.float32)
            d = dem.astype(np.float32)[None, ...]
            x = np.concatenate([x_rgb, d, d, d], axis=0)
        else:
            raise ValueError(f"Unknown input_mode: {self.input_mode}")

        x_hwc = np.transpose(x, (1, 2, 0))
        x_hwc = cv2.resize(x_hwc, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR)
        x = np.transpose(x_hwc, (2, 0, 1))
        y = cv2.resize((mask > 0).astype(np.float32), (self.resize_to, self.resize_to), interpolation=cv2.INTER_NEAREST)

        x = _minmax_per_channel(torch.from_numpy(x).float())
        y = torch.from_numpy(y[None, ...]).float()
        if self.transform is not None:
            x, y = self.transform(x, y)
        return {"image": x, "mask": y}


class BijieRawDataset(Dataset):
    def __init__(self, root_dir: str | Path, phase: str):
        self.root = Path(root_dir)
        self.phase = phase
        self.image_dir = self.root / "image"
        self.dem_dir = self.root / "dem"
        self.mask_dir = None if phase == "non-landslide" else (self.root / "mask")
        self.files = sorted([p for p in self.image_dir.glob("*.png")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        img_fp = self.files[idx]
        name = img_fp.stem
        image = np.array(Image.open(img_fp))
        dem = np.array(Image.open(self.dem_dir / f"{name}.png"))
        if self.mask_dir is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = np.array(Image.open(self.mask_dir / f"{name}.png"))
        return {"image": image, "dem": dem, "mask": mask}


class BijieTwoComposites(Dataset):
    """
    Dual-stream Bijie dataset:
      stream_a = RGB (3ch)
      stream_b = DEM repeated to 3ch
    """

    def __init__(
        self,
        base_dataset: Dataset,
        resize_to: int = 256,
        transform: Optional[AugmentDual2D] = None,
    ):
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

        xa = np.transpose(img, (2, 0, 1)).astype(np.float32)
        d = dem.astype(np.float32)[None, ...]
        xb = np.repeat(d, repeats=3, axis=0)
        y = (mask > 0).astype(np.float32)

        xa_hwc = np.transpose(xa, (1, 2, 0))
        xb_hwc = np.transpose(xb, (1, 2, 0))
        xa_hwc = cv2.resize(xa_hwc, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR)
        xb_hwc = cv2.resize(xb_hwc, (self.resize_to, self.resize_to), interpolation=cv2.INTER_LINEAR)
        xa = np.transpose(xa_hwc, (2, 0, 1))
        xb = np.transpose(xb_hwc, (2, 0, 1))
        y = cv2.resize(y, (self.resize_to, self.resize_to), interpolation=cv2.INTER_NEAREST)

        xa = _minmax_per_channel(torch.from_numpy(xa).float())
        xb = _minmax_per_channel(torch.from_numpy(xb).float())
        y = torch.from_numpy(y[None, ...]).float()

        if self.transform is not None:
            xa, xb, y = self.transform(xa, xb, y)
        return {"stream_a": xa, "stream_b": xb, "mask": y}


def build_l4s_split(dataset_root: str | Path, val_ratio: float = 0.1, seed: int = 42):
    img_dir = Path(dataset_root) / "TrainData" / "img"
    files = sorted(img_dir.glob("*.h5"))
    n = len(files)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, n - 1)
    val_ids = perm[:n_val]
    train_ids = perm[n_val:]
    return train_ids, val_ids


def build_bijie_split(dataset_root: str | Path, seed: int = 42):
    root = Path(dataset_root)
    # Accept either:
    #   .../Bijie-landslide-dataset
    # or parent dir:
    #   .../dataset_bijie_landslide (containing Bijie-landslide-dataset/)
    if not (root / "landslide").exists() and (root / "Bijie-landslide-dataset").exists():
        root = root / "Bijie-landslide-dataset"

    landslide = BijieRawDataset(root / "landslide", phase="landslide")
    nonlandslide = BijieRawDataset(root / "non-landslide", phase="non-landslide")
    if len(landslide) == 0 or len(nonlandslide) == 0:
        raise ValueError(
            f"Bijie dataset appears empty at {root}. Expected folders: "
            f"{root / 'landslide' / 'image'} and {root / 'non-landslide' / 'image'} with PNG files."
        )
    g = torch.Generator().manual_seed(seed)

    def _split(ds):
        n = len(ds)
        sizes = [int(0.7 * n), int(0.2 * n)]
        sizes.append(n - sum(sizes))
        return random_split(ds, sizes, generator=g)

    tl, vl, sl = _split(landslide)
    tn, vn, sn = _split(nonlandslide)
    train_raw = ConcatDataset([tl, tn])
    val_raw = ConcatDataset([vl, vn])
    test_raw = ConcatDataset([sl, sn])
    return train_raw, val_raw, test_raw
