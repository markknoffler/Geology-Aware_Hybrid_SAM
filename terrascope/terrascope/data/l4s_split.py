"""
Landslide4Sense splits must use masked tiles under TrainData/img + TrainData/mask only.

The official Landslide4Sense benchmark also ships val/test HDF5 subsets that often have no
matching mask labels locally; supervised training evaluation should therefore derive **train /
hold-out** subsets only from the labeled TrainData pool.

Default: deterministic **90% train / 10% hold-out**. The hold-out subset is disjoint from train
(indices never duplicated); no optimization step uses hold-out batches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch


def list_l4s_masked_img_paths(dataset_root: str | Path):
    root = Path(dataset_root)
    img_dir = root / "TrainData" / "img"
    mask_dir = root / "TrainData" / "mask"
    files = sorted(img_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No Landslide4Sense HDF5 patches under {img_dir}")
    return files, mask_dir


def build_l4s_split(
    dataset_root: str | Path,
    *,
    seed: int = 42,
    train_fraction: float = 0.9,
    holdout_fraction: float | None = None,
) -> Tuple[list[int], list[int]]:
    """
    Return (train_indices, holdout_indices) into **sorted** TrainData/*.h5 list.

    `holdout_fraction` overrides the complement if set (defaults to ``1 - train_fraction``).

    Naming: "hold-out" tiles are reserved for metrics / checkpoint selection only.
    Training DataLoader samples **only** from ``train_indices``.
    """
    files, _ = list_l4s_masked_img_paths(dataset_root)
    n = len(files)
    if n < 10:
        raise ValueError(f"Expected many labeled HDF5 patches, got only {n} under TrainData.")

    tf = float(train_fraction)
    if not (0.0 < tf < 1.0):
        raise ValueError("train_fraction must be in (0, 1).")
    ho = float(1.0 - tf) if holdout_fraction is None else float(holdout_fraction)
    if ho <= 0 or ho >= 1:
        raise ValueError("hold-out fraction invalid.")

    g = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=g).tolist()
    n_hold = max(1, int(round(n * ho)))
    n_hold = min(n_hold, n - 1)

    holdout_ids = sorted(perm[:n_hold])
    train_ids = sorted(perm[n_hold:])
    assert not set(train_ids) & set(holdout_ids)
    return train_ids, holdout_ids
