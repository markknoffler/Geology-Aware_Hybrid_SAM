from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


EPSILON = 1e-6


def _read_named_or_first(path: Path, preferred_key: str) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if preferred_key in handle:
            return np.asarray(handle[preferred_key], dtype=np.float32)
        for key in handle.keys():
            value = np.asarray(handle[key], dtype=np.float32)
            if value.ndim >= 2:
                return value
    raise ValueError(f"No readable array found in {path}")


class LandSlide4Sense(Dataset):
    def __init__(self, split_dir: str):
        self.split_dir = Path(split_dir)
        self.images = sorted((self.split_dir / "img").glob("*.h5"))
        self.masks = sorted((self.split_dir / "mask").glob("*.h5"))
        self.has_mask = len(self.masks) == len(self.images) and len(self.images) > 0

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        img = _read_named_or_first(img_path, "img")
        img[np.isnan(img)] = EPSILON
        if img.ndim == 3 and img.shape[0] > 20:
            img = img.transpose((2, 0, 1))
        elif img.ndim == 3 and img.shape[-1] <= 20:
            img = img.transpose((2, 0, 1))

        if self.has_mask:
            mask_path = self.masks[idx]
            mask = _read_named_or_first(mask_path, "mask")
        else:
            mask = np.zeros(img.shape[-2:], dtype=np.float32)

        return img.astype(np.float32), mask.astype(np.float32), img_path.stem


def add_gaussian_noise(tensor_img: torch.Tensor, mean: float = 0.0, std: float = 0.05) -> torch.Tensor:
    noise = torch.randn_like(tensor_img) * std + mean
    return tensor_img + noise


def add_salt_pepper_noise(
    tensor_img: torch.Tensor, amount: float = 0.005, salt_vs_pepper: float = 0.5
) -> torch.Tensor:
    del salt_vs_pepper
    img = tensor_img.clone().detach().cpu().numpy()
    channels, height, width = img.shape
    num_pixels = int(amount * height * width)

    for idx in range(channels):
        coords = [np.random.randint(0, height, num_pixels), np.random.randint(0, width, num_pixels)]
        img[idx][tuple(coords)] = 1.0
        coords = [np.random.randint(0, height, num_pixels), np.random.randint(0, width, num_pixels)]
        img[idx][tuple(coords)] = 0.0

    return torch.from_numpy(img).type_as(tensor_img)


def apply_clahe(
    tensor_img: torch.Tensor, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)
) -> torch.Tensor:
    img = tensor_img.clone().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    for idx in range(img.shape[2]):
        img[:, :, idx] = clahe.apply(img[:, :, idx])

    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img).type_as(tensor_img)


class DualStreamTransform:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image1: torch.Tensor, image2: torch.Tensor, label: torch.Tensor):
        if np.random.rand() < self.p:
            image1 = TF.hflip(image1)
            image2 = TF.hflip(image2)
            label = TF.hflip(label)

        if np.random.rand() < self.p:
            image1 = TF.vflip(image1)
            image2 = TF.vflip(image2)
            label = TF.vflip(label)

        if np.random.rand() < self.p:
            image1 = add_gaussian_noise(image1)
            image2 = add_gaussian_noise(image2)

        if np.random.rand() < self.p:
            image1 = add_salt_pepper_noise(image1)
            image2 = add_salt_pepper_noise(image2)

        if np.random.rand() < self.p:
            image1 = apply_clahe(image1)
            image2 = apply_clahe(image2)

        return image1, image2, label


class Landslide4SenseDualStream(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        bands: str = "RGB-NDVI-SLOPE-DEM",
        resize_to: Optional[int | Tuple[int, int]] = 256,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        split_map = {"train": "TrainData", "valid": "ValidData", "val": "ValidData", "test": "TestData"}
        if split.lower() not in split_map:
            raise ValueError(f"Unknown split: {split}")

        self.dataset = LandSlide4Sense(str(Path(data_root) / split_map[split.lower()]))
        self.bands = bands
        self.transform = transform
        self.has_mask = self.dataset.has_mask
        self.resize_to = (resize_to, resize_to) if isinstance(resize_to, int) else resize_to
        self.band_map = {"B2": 1, "B3": 2, "B4": 3, "B8": 7, "B8A": 8, "B11": 10, "B12": 11, "B13": 12, "B14": 13}

    def __len__(self) -> int:
        return len(self.dataset)

    def _create_composite(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        b2 = image[self.band_map["B2"]]
        b3 = image[self.band_map["B3"]]
        b4 = image[self.band_map["B4"]]
        b8 = image[self.band_map["B8"]]
        b8a = image[self.band_map["B8A"]]
        b11 = image[self.band_map["B11"]]
        b12 = image[self.band_map["B12"]]
        b13 = image[self.band_map["B13"]]
        b14 = image[self.band_map["B14"]]

        if self.bands in {"RGB-NSE", "RGB-NDVI-SLOPE-DEM"}:
            ndvi = np.clip((b8 - b4) / (b8 + b4 + EPSILON), -1, 1)
            comp1 = np.stack([b4, b3, b2], axis=0)
            comp2 = np.stack([ndvi, b13, b14], axis=0)
            return comp1, comp2
        if self.bands == "RGB&DEM":
            return np.stack([b4, b3, b2], axis=0), np.stack([b14, b14, b14], axis=0)
        if self.bands == "RGB&SWIR":
            return np.stack([b4, b3, b2], axis=0), np.stack([b11, b12, b8a], axis=0)
        raise ValueError(f"Composite Error: '{self.bands}'")

    def _normalize(self, image_tensor: torch.Tensor) -> torch.Tensor:
        for idx in range(image_tensor.shape[0]):
            channel = image_tensor[idx]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                channel = (channel - min_val) / (max_val - min_val + EPSILON)
            image_tensor[idx] = torch.clamp(channel, 0.0, 1.0)
        return image_tensor

    def _resize_image(self, img: np.ndarray) -> np.ndarray:
        img = np.transpose(img, (1, 2, 0))
        img = cv2.resize(img, self.resize_to, interpolation=cv2.INTER_LINEAR)
        return np.transpose(img, (2, 0, 1))

    def __getitem__(self, idx: int):
        image, label, file_name = self.dataset[idx]
        comp1, comp2 = self._create_composite(image)

        if self.resize_to is not None:
            comp1 = self._resize_image(comp1)
            comp2 = self._resize_image(comp2)
            label = cv2.resize(label, self.resize_to, interpolation=cv2.INTER_NEAREST)

        comp1_tensor = self._normalize(torch.from_numpy(comp1).float())
        comp2_tensor = self._normalize(torch.from_numpy(comp2).float())
        label_tensor = torch.from_numpy((label > 0).astype(np.float32))

        if self.transform is not None:
            comp1_tensor, comp2_tensor, label_tensor = self.transform(comp1_tensor, comp2_tensor, label_tensor)

        return {
            "stream_a": comp1_tensor,
            "stream_b": comp2_tensor,
            "mask": label_tensor.unsqueeze(0),
            "id": file_name,
        }
