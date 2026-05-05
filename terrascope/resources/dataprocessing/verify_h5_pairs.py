#!/usr/bin/env python3
"""Verify Landslide4Sense-style .h5 image/mask pairing under a root directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from terrascope.data.l4s_h5_dataset import pair_h5_files


def main():
    p = argparse.ArgumentParser(description="List and validate H5 image/mask pairs")
    p.add_argument("--images", type=Path, required=True, help="Directory of image .h5 files")
    p.add_argument("--masks", type=Path, default=None, help="Directory of mask .h5 files (optional)")
    args = p.parse_args()
    pairs = pair_h5_files(args.images, args.masks)
    print(f"Found {len(pairs)} pairs.")
    for img, m in pairs[:20]:
        print(f"  {img.name} -> {m.name if m else 'none'}")
    if len(pairs) > 20:
        print("  ...")


if __name__ == "__main__":
    main()
