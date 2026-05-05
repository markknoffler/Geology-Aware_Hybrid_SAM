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
    raise ValueError(f"No valid arrays found in {path}")


def summarize_split(split_dir: Path):
    img_dir = split_dir / "img"
    mask_dir = split_dir / "mask"
    img_files = sorted(img_dir.glob("*.h5"))
    has_mask = mask_dir.exists()

    channel_counts = []
    shapes = {}
    mask_positive_ratio = []

    for p in tqdm(img_files, desc=f"Inspecting {split_dir.name}"):
        arr = read_h5(p)
        if arr.ndim == 2:
            c, h, w = 1, arr.shape[0], arr.shape[1]
        elif arr.ndim == 3:
            if arr.shape[0] <= 20:
                c, h, w = arr.shape
            else:
                h, w, c = arr.shape
        else:
            continue

        channel_counts.append(int(c))
        shape_key = f"{h}x{w}"
        shapes[shape_key] = shapes.get(shape_key, 0) + 1

        if has_mask:
            image_id = p.stem.replace("image_", "")
            mask_path = mask_dir / f"mask_{image_id}.h5"
            if mask_path.exists():
                m = read_h5(mask_path)
                if m.ndim == 3:
                    m = m[0] if m.shape[0] == 1 else m[..., 0]
                mask_positive_ratio.append(float((m > 0).mean()))

    out = {
        "num_images": len(img_files),
        "has_masks": has_mask,
        "channel_count_min": int(min(channel_counts)) if channel_counts else 0,
        "channel_count_max": int(max(channel_counts)) if channel_counts else 0,
        "spatial_shapes": shapes,
        "mask_positive_ratio_mean": float(np.mean(mask_positive_ratio)) if mask_positive_ratio else None,
    }
    return out


def main():
    p = argparse.ArgumentParser(description="Inspect and summarize Landslide4Sense H5 dataset structure.")
    p.add_argument("--dataset_root", type=str, required=True)
    p.add_argument("--output_json", type=str, default="dataset_summary.json")
    args = p.parse_args()

    root = Path(args.dataset_root)
    splits = ["TrainData", "ValidData", "TestData"]
    summary = {}

    for s in splits:
        split_dir = root / s
        if split_dir.exists():
            summary[s] = summarize_split(split_dir)

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {out_path}")


if __name__ == "__main__":
    main()
