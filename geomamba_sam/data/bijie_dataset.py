from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


@dataclass
class BijieConfig:
    root: str
    target_size: int = 512
    split_seed: int = 42
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    dem_norm: str = "zscore"  # zscore|minmax


def _collect_samples(root: Path) -> List[Dict]:
    base = root / "Bijie-landslide-dataset"
    ls_image = base / "landslide" / "image"
    ls_dem = base / "landslide" / "dem"
    ls_mask = base / "landslide" / "mask"
    nls_image = base / "non-landslide" / "image"
    nls_dem = base / "non-landslide" / "dem"

    out = []
    for img in sorted(ls_image.glob("*.png")):
        stem = img.stem
        dem = ls_dem / f"{stem}.png"
        mask = ls_mask / f"{stem}.png"
        if not dem.exists() or not mask.exists():
            raise FileNotFoundError(f"Missing DEM/mask for landslide sample: {stem}")
        out.append({"id": stem, "image": img, "dem": dem, "mask": mask, "label": 1})

    for img in sorted(nls_image.glob("*.png")):
        stem = img.stem
        dem = nls_dem / f"{stem}.png"
        if not dem.exists():
            raise FileNotFoundError(f"Missing DEM for non-landslide sample: {stem}")
        out.append({"id": stem, "image": img, "dem": dem, "mask": None, "label": 0})
    return out


def _load_official_split_ids(root: Path):
    candidates = [
        root / "Bijie-landslide-dataset" / "splits",
        root / "splits",
    ]
    for base in candidates:
        train_f = base / "train.txt"
        val_f = base / "val.txt"
        test_f = base / "test.txt"
        if train_f.exists() and val_f.exists() and test_f.exists():
            def read_ids(path: Path):
                return {ln.strip().split(".")[0] for ln in path.read_text().splitlines() if ln.strip()}
            return read_ids(train_f), read_ids(val_f), read_ids(test_f)
    return None


class BijieDataset(Dataset):
    """
    For non-landslide class, returns all-black mask as requested.
    Uses deterministic stratified split with 70/20/10 default.
    """

    def __init__(self, config: BijieConfig, split: str = "train"):
        self.cfg = config
        self.split = split
        samples = _collect_samples(Path(config.root))
        official = _load_official_split_ids(Path(config.root))

        if official is not None:
            train_ids, val_ids, test_ids = official
            id_map = {"train": train_ids, "val": val_ids, "test": test_ids}
            wanted = id_map[split]
            self.samples = [s for s in samples if s["id"] in wanted]
            if not self.samples:
                raise ValueError(f"Official split '{split}' resolved to empty set.")
            return

        pos = [s for s in samples if s["label"] == 1]
        neg = [s for s in samples if s["label"] == 0]
        rng = np.random.RandomState(config.split_seed)
        rng.shuffle(pos)
        rng.shuffle(neg)

        def split_three(arr):
            n = len(arr)
            n_train = int(n * config.train_ratio)
            n_val = int(n * config.val_ratio)
            n_test = n - n_train - n_val
            return arr[:n_train], arr[n_train:n_train + n_val], arr[n_train + n_val:n_train + n_val + n_test]

        p_train, p_val, p_test = split_three(pos)
        n_train, n_val, n_test = split_three(neg)
        splits = {
            "train": p_train + n_train,
            "val": p_val + n_val,
            "test": p_test + n_test,
        }
        self.samples = splits[split]
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _read_rgb(self, path: Path):
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)

    def _read_dem(self, path: Path):
        return np.asarray(Image.open(path).convert("L"), dtype=np.float32)

    def _read_mask(self, sample: Dict, h: int, w: int):
        if sample["mask"] is None:
            return np.zeros((h, w), dtype=np.float32)
        return (np.asarray(Image.open(sample["mask"]).convert("L"), dtype=np.float32) > 127).astype(np.float32)

    def __getitem__(self, idx):
        s = self.samples[idx]
        rgb = self._read_rgb(s["image"])
        dem = self._read_dem(s["dem"])
        mask = self._read_mask(s, rgb.shape[0], rgb.shape[1])

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1)
        dem_t = torch.from_numpy(dem).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        rgb_t = TF.resize(rgb_t, [self.cfg.target_size, self.cfg.target_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        dem_t = TF.resize(dem_t, [self.cfg.target_size, self.cfg.target_size], interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask_t = TF.resize(mask_t, [self.cfg.target_size, self.cfg.target_size], interpolation=InterpolationMode.NEAREST)

        rgb_t = TF.normalize(rgb_t / 255.0, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if self.cfg.dem_norm == "minmax":
            dem_min, dem_max = dem_t.min(), dem_t.max()
            dem_t = (dem_t - dem_min) / (dem_max - dem_min + 1e-6)
        else:
            dem_t = (dem_t - dem_t.mean()) / (dem_t.std() + 1e-6)
        mask_t = (mask_t > 0.5).float()
        return rgb_t.float(), dem_t.float(), mask_t.float(), {"id": s["id"], "label": s["label"]}
