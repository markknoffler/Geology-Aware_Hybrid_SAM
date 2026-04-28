# my_dataset.py
import glob, os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class SegDataset(Dataset):
    """
    Loads <image>.png and <mask>.png, keeps aspect ratio,
    then pads (or optionally resizes) so the final canvas
    is exactly 512 × 512.
    """
    def __init__(self, img_dir: str, mask_dir: str):
        self.imgs  = sorted(Path(img_dir).glob("*.png"))
        self.masks = sorted(Path(mask_dir).glob("*.png"))
        assert len(self.imgs) == len(self.masks), "image / mask count differs"
        self.target = 512      # fixed edge length

    def __len__(self): return len(self.imgs)

    def __getitem__(self, idx):
        img  = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")

        # ---------- preserve aspect ratio, short side → 512 ----------
        w, h  = img.size
        scale = self.target / min(w, h)
        nw, nh = int(w * scale), int(h * scale)

        img  = TF.resize(img,  (nh, nw), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (nh, nw), interpolation=InterpolationMode.NEAREST)

        # ---------- pad to 512 × 512 (right / bottom) ----------------
        pad_w = self.target - nw
        pad_h = self.target - nh
        img  = TF.pad(img,  (0, 0, pad_w, pad_h), fill=0)
        mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=0)

        # ---------- tensor + normalise --------------------------------
        img  = TF.to_tensor(img)
        img  = TF.normalize(img,
                            mean=[0.485, 0.456, 0.406],
                            std =[0.229, 0.224, 0.225])
        mask = (TF.to_tensor(mask) > 0.5).float()      # 1×512×512
        return img, mask
