from terrascope.data.bijie_dataset import BijieConfig, BijieDataset
from terrascope.data.l4s_h5_dataset import L4SConfig, Landslide4SenseH5Dataset
from terrascope.data.l4s_split import build_l4s_split


def build_datasets(
    dataset_name: str,
    dataset_root: str,
    target_size: int,
    seed: int,
    *,
    l4s_train_fraction: float = 0.9,
):
    dataset_name = dataset_name.lower()
    if dataset_name == "landslide4sense":
        # Supervised folds only use TrainData/*/ with masks — never unlabeled Challenge val/test HDF5 drops.
        train_idx, holdout_idx = build_l4s_split(dataset_root, seed=seed, train_fraction=l4s_train_fraction)
        train_ds = Landslide4SenseH5Dataset(
            L4SConfig(root=dataset_root, target_size=target_size, augment_train=True),
            train_idx,
            split="train",
        )
        val_ds = Landslide4SenseH5Dataset(
            L4SConfig(root=dataset_root, target_size=target_size, augment_train=False),
            holdout_idx,
            split="holdout",
        )
        test_ds = None  # reuse hold-out Loader in train.py final row (same 10%).
        return train_ds, val_ds, test_ds

    if dataset_name == "bijie":
        cfg = BijieConfig(root=dataset_root, target_size=target_size, split_seed=seed)
        train_ds = BijieDataset(cfg, split="train")
        val_ds = BijieDataset(cfg, split="val")
        test_ds = BijieDataset(cfg, split="test")
        return train_ds, val_ds, test_ds

    raise ValueError(f"Unsupported dataset '{dataset_name}'. Use: landslide4sense or bijie.")
