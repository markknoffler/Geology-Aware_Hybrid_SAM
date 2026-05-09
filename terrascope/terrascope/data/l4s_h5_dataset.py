"""
Landslide4Sense HDF5 loader aligned with the DiGATe-style dual-stream baseline:
RGB = (B4, B3, B2), Stream B (topography) = (NDVI from B8/B4, slope B13, DEM B14).

See Landslide4Sense paper / IEEE TGRS format and SAM/ablation_study/baseline_models/common/datasets.py.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

_EPS = 1e-6
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _pair_h5_files(img_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    img_files = sorted(img_dir.glob("*.h5"))
    if not img_files:
        raise FileNotFoundError(f"No .h5 files found in {img_dir}")

    masks_by_suffix = {}
    for m in sorted(mask_dir.glob("*.h5")):
        masks_by_suffix[m.stem.split("_")[-1]] = m

    pairs = []
    for img in img_files:
        key = img.stem.split("_")[-1]
        if key not in masks_by_suffix:
            raise ValueError(f"Missing mask pair for image {img.name}")
        pairs.append((img, masks_by_suffix[key]))
    return pairs


def _read_first_existing(hf: h5py.File, candidates: Sequence[str]) -> np.ndarray:
    for k in candidates:
        if k in hf:
            return np.asarray(hf[k])
    if len(hf.keys()) == 1:
        return np.asarray(hf[next(iter(hf.keys()))])
    raise KeyError(f"No known key in h5 ({list(hf.keys())})")


def _to_band_chw(raw: np.ndarray) -> np.ndarray:
    """Normalize Landslide4Sense cube to CHW (~14 Sentinel bands per Baseline loaders)."""
    image = np.asarray(raw, dtype=np.float32).squeeze()
    if image.ndim == 3 and image.shape[0] > 20:
        pass  # already CHW
    elif image.ndim == 3 and image.shape[-1] <= 20:
        image = np.transpose(image, (2, 0, 1))
    elif image.ndim == 2:
        image = image[None, ...]
    elif image.ndim == 3:
        image = np.transpose(image, (2, 0, 1))
    else:
        raise ValueError(f"Unexpected ndarray shape after squeeze {image.shape}")
    return image


def _to_binary_mask(mask: np.ndarray, *, threshold_zero: float = 1e-6) -> np.ndarray:
    m = np.squeeze(mask.astype(np.float32))
    return (np.abs(m) > threshold_zero).astype(np.float32)


def _minmax_chw_tensor(t: torch.Tensor) -> torch.Tensor:
    out = t.clone()
    for c in range(out.shape[0]):
        plane = out[c]
        mn = plane.min().item()
        mx = plane.max().item()
        if mx > mn:
            plane = (plane - mn) / (mx - mn + _EPS)
        out[c] = torch.clamp(plane, 0.0, 1.0)
    return out


@dataclass
class L4SConfig:
    root: str
    image_key_candidates: Tuple[str, ...] = ("img", "image", "images", "x", "X")
    mask_key_candidates: Tuple[str, ...] = ("mask", "masks", "label", "labels", "y", "Y")
    target_size: int = 256
    augment_train: bool = True


class Landslide4SenseH5Dataset(Dataset):
    """
    Paper-aligned dual modalities from one H5 multispectral cube.
    Optionally uses list of indices into sorted TrainData img list (paired with masks).
    """

    def __init__(
        self,
        config: L4SConfig,
        indices: Optional[Sequence[int]] = None,
        *,
        split: str = "train",
    ):
        self.cfg = config
        self.split = split
        root = Path(config.root)
        train_img = root / "TrainData" / "img"
        train_mask = root / "TrainData" / "mask"
        all_pairs = _pair_h5_files(train_img, train_mask)

        if indices is None:
            idx_range = np.arange(len(all_pairs))
        else:
            idx_range = np.asarray(list(indices), dtype=np.int64)
        self.pairs = [all_pairs[i] for i in idx_range]

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _spectral_to_rgb_topo(image_chw: np.ndarray):
        """Match baseline_models L4SDualStreamDataset indexing."""
        if image_chw.shape[0] < 14:
            raise ValueError(
                f"Landslide4Sense patch needs ≥14 Sentinel-2 stacked bands (CHW), got C={image_chw.shape[0]}"
            )
        B2, B3, B4 = image_chw[1], image_chw[2], image_chw[3]
        B8 = image_chw[7]
        B13, B14 = image_chw[12], image_chw[13]
        ndvi = np.clip((B8 - B4) / (B8 + B4 + _EPS), -1.0, 1.0).astype(np.float32)
        rgb = np.stack([B4, B3, B2], axis=0).astype(np.float32)
        topo = np.stack([ndvi, B13, B14], axis=0).astype(np.float32)
        return rgb, topo

    def _load_arrays(self, img_path: Path, mask_path: Path):
        with h5py.File(img_path, "r") as hf:
            raw = _read_first_existing(hf, self.cfg.image_key_candidates)
        with h5py.File(mask_path, "r") as hf:
            mraw = np.asarray(_read_first_existing(hf, self.cfg.mask_key_candidates))

        image_chw = _to_band_chw(raw)
        rgb, topo = self._spectral_to_rgb_topo(image_chw)
        mask = _to_binary_mask(mraw)
        if mask.ndim == 0:
            mask = np.zeros(rgb.shape[-2:], dtype=np.float32)
        return rgb, topo, mask

    def _maybe_augment(self, rgb_t: torch.Tensor, topo_t: torch.Tensor, mask_t: torch.Tensor):
        if not self.cfg.augment_train or self.split != "train":
            return rgb_t, topo_t, mask_t
        p = 0.5
        if random.random() < p:
            rgb_t = TF.hflip(rgb_t)
            topo_t = TF.hflip(topo_t)
            mask_t = TF.hflip(mask_t)
        if random.random() < p:
            rgb_t = TF.vflip(rgb_t)
            topo_t = TF.vflip(topo_t)
            mask_t = TF.vflip(mask_t)
        if random.random() < p:
            rgb_t = rgb_t + torch.randn_like(rgb_t) * 0.05
            topo_t = topo_t + torch.randn_like(topo_t) * 0.05
        topo_t = torch.clamp(topo_t, min=-3.0, max=3.0)
        return rgb_t, topo_t, mask_t

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        rgb, topo, mask = self._load_arrays(img_path, mask_path)

        rgb_t = torch.from_numpy(rgb)
        topo_t = torch.from_numpy(topo)
        mask_t = torch.from_numpy(mask[None, ...])

        hw = self.cfg.target_size
        rgb_t = TF.resize(rgb_t, [hw, hw], interpolation=InterpolationMode.BILINEAR, antialias=True)
        topo_t = TF.resize(topo_t, [hw, hw], interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask_t = TF.resize(mask_t, [hw, hw], interpolation=InterpolationMode.NEAREST)

        # 1. RGB Stream: ImageNet Normalization
        rgb_t = rgb_t / 10000.0  # Sentinel-2 raw to roughly [0, 1]
        rgb_t = torch.clamp(rgb_t, 0.0, 1.0)
        for c in range(3):
            rgb_t[c] = (rgb_t[c] - _IMAGENET_MEAN[c]) / (_IMAGENET_STD[c] + _EPS)

        # 2. Topography Stream: Global Physical Scaling (Preserve Absolute Features)
        # topo_t[0] = NDVI [-1, 1], topo_t[1] = Slope [0, 90], topo_t[2] = DEM [0, 10000]
        ndvi = (topo_t[0:1] + 1.0) * 0.5  # [-1, 1] -> [0, 1]
        slope = topo_t[1:2] / 90.0        # [0, 90] -> [0, 1]
        dem = topo_t[2:3] / 10000.0       # [0, 10000] -> [0, 1]
        topo_t = torch.cat([ndvi, slope, dem], dim=0)
        topo_t = torch.clamp(topo_t, 0.0, 1.0)

        mask_t = (mask_t > 0.5).float()

        rgb_t, topo_t, mask_t = self._maybe_augment(rgb_t, topo_t, mask_t)
        return rgb_t.float(), topo_t.float(), mask_t.float(), {"id": img_path.stem}
