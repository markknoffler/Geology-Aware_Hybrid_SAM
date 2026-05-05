#!/usr/bin/env python3
"""Validate Bijie dataset folder layout expected by terrascope.data.bijie_dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from terrascope.data.bijie_dataset import collect_bijie_samples


def main():
    p = argparse.ArgumentParser(description="Check Bijie RGB/DEM/mask file triplets")
    p.add_argument("--root", type=Path, required=True, help="Dataset root containing Bijie-landslide-dataset/")
    args = p.parse_args()
    samples = collect_bijie_samples(args.root)
    print(f"OK: {len(samples)} samples under {args.root / 'Bijie-landslide-dataset'}")
    for s in samples[:5]:
        print(f"  {s['id']}: image={s['image'].name} dem={s['dem'].name} mask={s['mask'].name}")
    if len(samples) > 5:
        print("  ...")


if __name__ == "__main__":
    main()
