import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def read_h5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        for key in f.keys():
            arr = np.asarray(f[key])
            if arr.ndim >= 2:
                return arr
    raise ValueError(f"No valid arrays in {path}")


def summarize_split(split_dir: Path):
    img_dir = split_dir / "img"
    mask_dir = split_dir / "mask"
    img_files = sorted(img_dir.glob("*.h5"))
    has_mask = mask_dir.exists()

    channels = []
    shapes = {}
    positive_ratio = []
    for p in tqdm(img_files, desc=f"Inspecting {split_dir.name}"):
        x = read_h5(p)
        if x.ndim == 2:
            c, h, w = 1, x.shape[0], x.shape[1]
        elif x.ndim == 3 and x.shape[0] <= 20:
            c, h, w = x.shape
        elif x.ndim == 3:
            h, w, c = x.shape
        else:
            continue

        channels.append(int(c))
        key = f"{h}x{w}"
        shapes[key] = shapes.get(key, 0) + 1

        if has_mask:
            image_id = p.stem.replace("image_", "")
            mp = mask_dir / f"mask_{image_id}.h5"
            if mp.exists():
                m = read_h5(mp)
                if m.ndim == 3:
                    m = m[0] if m.shape[0] == 1 else m[..., 0]
                positive_ratio.append(float((m > 0).mean()))

    return {
        "num_images": len(img_files),
        "has_masks": has_mask,
        "channel_count_min": min(channels) if channels else 0,
        "channel_count_max": max(channels) if channels else 0,
        "spatial_shapes": shapes,
        "mask_positive_ratio_mean": float(np.mean(positive_ratio)) if positive_ratio else None,
    }


def main():
    p = argparse.ArgumentParser(description="Inspect Landslide4Sense dataset structure.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--output_json", type=str, default="dataset_summary.json")
    args = p.parse_args()

    root = Path(args.dataset_root)
    summary = {}
    for split in ["TrainData", "ValidData", "TestData"]:
        split_dir = root / split
        if split_dir.exists():
            summary[split] = summarize_split(split_dir)

    out = Path(args.output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out}")


if __name__ == "__main__":
    main()
