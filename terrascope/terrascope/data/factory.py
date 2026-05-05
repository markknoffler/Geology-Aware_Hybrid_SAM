from terrascope.data.bijie_dataset import BijieConfig, BijieDataset
from terrascope.data.l4s_h5_dataset import L4SConfig, Landslide4SenseH5Dataset


def build_datasets(dataset_name: str, dataset_root: str, target_size: int, seed: int):
    dataset_name = dataset_name.lower()
    if dataset_name == "landslide4sense":
        cfg = L4SConfig(root=dataset_root, target_size=target_size, split_seed=seed)
        train_ds = Landslide4SenseH5Dataset(cfg, split="train")
        test_ds = Landslide4SenseH5Dataset(cfg, split="test")
        return train_ds, None, test_ds

    if dataset_name == "bijie":
        cfg = BijieConfig(root=dataset_root, target_size=target_size, split_seed=seed)
        train_ds = BijieDataset(cfg, split="train")
        val_ds = BijieDataset(cfg, split="val")
        test_ds = BijieDataset(cfg, split="test")
        return train_ds, val_ds, test_ds

    raise ValueError(f"Unsupported dataset '{dataset_name}'. Use: landslide4sense or bijie.")
