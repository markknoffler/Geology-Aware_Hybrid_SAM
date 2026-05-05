from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_h5_array(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arr = np.asarray(f[key])
            if arr.ndim >= 2:
                return arr
    raise ValueError(f"No readable array found in {path}")


def _to_chw(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[None, ...]
    if image.ndim == 3:
        if image.shape[0] <= 20:
            return image
        return np.transpose(image, (2, 0, 1))
    raise ValueError(f"Unsupported image shape: {image.shape}")


def _minmax_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mn = float(x.min())
    mx = float(x.max())
    return (x - mn) / (mx - mn + eps)


class Landslide4SenseBiFusion(Dataset):
    """
    Optical-only loader by default (paper setting).
    Uses TrainData/img + TrainData/mask and receives explicit file subset.
    """

    def __init__(
        self,
        image_paths: List[Path],
        mask_dir: Path,
        rgb_indices: Tuple[int, int, int] = (3, 2, 1),
    ):
        super().__init__()
        self.image_paths = sorted(image_paths)
        self.mask_dir = mask_dir
        self.rgb_indices = rgb_indices

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image_path = self.image_paths[idx]
        img = _to_chw(_read_h5_array(image_path)).astype(np.float32)
        channels, height, width = img.shape

        needed = list(self.rgb_indices)
        if max(needed) >= channels:
            raise ValueError(
                f"Insufficient channels in {image_path}. Needed up to index {max(needed)} but got {channels} channels."
            )

        x = img[list(self.rgb_indices)]
        x = _minmax_norm(x).astype(np.float32)

        image_id = image_path.stem.replace("image_", "")
        mask_path = self.mask_dir / f"mask_{image_id}.h5"
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file not found for {image_path}: {mask_path}")
        mask = _read_h5_array(mask_path)
        if mask.ndim == 3:
            mask = mask[0] if mask.shape[0] == 1 else mask[..., 0]
        mask = (mask > 0).astype(np.float32)

        return {
            "image": torch.from_numpy(x).float(),
            "mask": torch.from_numpy(mask[None, ...]).float(),
            "id": image_path.stem,
            "shape_hw": torch.tensor([height, width], dtype=torch.int32),
        }
