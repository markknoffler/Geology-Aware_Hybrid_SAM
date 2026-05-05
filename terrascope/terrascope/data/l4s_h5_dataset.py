from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


def _pair_h5_files(img_dir: Path, mask_dir: Optional[Path]) -> List[Tuple[Path, Optional[Path]]]:
    img_files = sorted(img_dir.glob("*.h5"))
    if not img_files:
        raise FileNotFoundError(f"No .h5 files found in {img_dir}")
    if mask_dir is None:
        return [(p, None) for p in img_files]

    masks_by_suffix = {}
    for m in sorted(mask_dir.glob("*.h5")):
        key = m.stem.split("_")[-1]
        masks_by_suffix[key] = m

    pairs = []
    for img in img_files:
        key = img.stem.split("_")[-1]
        if key not in masks_by_suffix:
            raise ValueError(f"Missing mask pair for image {img.name}")
        pairs.append((img, masks_by_suffix[key]))
    return pairs


pair_h5_files = _pair_h5_files


def _read_first_existing(hf: h5py.File, candidates: Sequence[str]) -> np.ndarray:
    for k in candidates:
        if k in hf:
            return np.asarray(hf[k])
    if len(hf.keys()) == 1:
        return np.asarray(hf[next(iter(hf.keys()))])
    raise KeyError(f"Unable to detect key in h5 file. candidates={candidates}, available={list(hf.keys())}")


@dataclass
class L4SConfig:
    root: str
    image_key_candidates: Tuple[str, ...] = ("img", "image", "images", "x", "X")
    mask_key_candidates: Tuple[str, ...] = ("mask", "masks", "label", "labels", "y", "Y")
    dem_channel: int = 3
    rgb_channels: Tuple[int, int, int] = (0, 1, 2)
    target_size: int = 512
    split_seed: int = 42


class Landslide4SenseH5Dataset(Dataset):
    """
    Uses TrainData/img + TrainData/mask and creates a deterministic split:
    - train: 90%
    - test: 10%
    """

    def __init__(self, config: L4SConfig, split: str = "train"):
        self.cfg = config
        self.split = split
        root = Path(config.root)
        train_img = root / "TrainData" / "img"
        train_mask = root / "TrainData" / "mask"
        self.pairs = _pair_h5_files(train_img, train_mask)

        rng = np.random.RandomState(config.split_seed)
        idx = np.arange(len(self.pairs))
        rng.shuffle(idx)
        cut = int(0.9 * len(idx))
        train_idx = idx[:cut]
        test_idx = idx[cut:]
        chosen = train_idx if split == "train" else test_idx
        self.pairs = [self.pairs[i] for i in chosen]

    def __len__(self):
        return len(self.pairs)

    def _load_modalities(self, img_path: Path, mask_path: Path):
        with h5py.File(img_path, "r") as hf:
            arr = _read_first_existing(hf, self.cfg.image_key_candidates)
        with h5py.File(mask_path, "r") as hf:
            m = _read_first_existing(hf, self.cfg.mask_key_candidates)

        arr = np.squeeze(arr)
        m = np.squeeze(m)
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.shape[-1] <= self.cfg.dem_channel:
            raise ValueError(f"DEM channel index {self.cfg.dem_channel} out of range for {img_path.name}: {arr.shape}")

        rgb = arr[..., list(self.cfg.rgb_channels)].astype(np.float32)
        dem = arr[..., self.cfg.dem_channel].astype(np.float32)
        mask = m.astype(np.float32)
        return rgb, dem, mask

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        rgb, dem, mask = self._load_modalities(img_path, mask_path)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        dem_t = torch.from_numpy(dem).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        rgb_t = TF.resize(rgb_t, [self.cfg.target_size, self.cfg.target_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        dem_t = TF.resize(dem_t, [self.cfg.target_size, self.cfg.target_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask_t = TF.resize(mask_t, [self.cfg.target_size, self.cfg.target_size], interpolation=InterpolationMode.NEAREST)

        rgb_t = TF.normalize(rgb_t / (rgb_t.max().clamp(min=1e-6)), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        dem_t = (dem_t - dem_t.mean()) / (dem_t.std().clamp(min=1e-6))
        mask_t = (mask_t > 0.5).float()
        return rgb_t.float(), dem_t.float(), mask_t.float(), {"id": img_path.stem}
