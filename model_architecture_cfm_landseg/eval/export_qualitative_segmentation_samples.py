#!/usr/bin/env python3
"""
Export qualitative segmentation samples: RGB image + GT mask + predicted mask.

Uses TriEncoderCFMNet checkpoints (default: runs/*/tri_encoder_cfm_v2/checkpoint/best.pt).

Three folders, 11 samples each (default):
  bijie_landslide/       — Bijie landslide/ tiles (GT masks present)
  bijie_non_landslide/   — Bijie non-landslide/ tiles (GT = empty)
  landslide4sense/       — Landslide4Sense TrainData H5 tiles

Per sample (same stem):
  {stem}_image.png
  {stem}_mask_gt.png
  {stem}_mask_pred.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Subset

_CODE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_code_root() -> Path:
    here = Path(__file__).resolve()
    for base in (here.parents[1], here.parents[2]):
        if (base / "tri_cfm_net.py").is_file():
            return base
        if (base.parent / "tri_cfm_net.py").is_file():
            return base.parent
    return here.parents[1]


_CODE_ROOT = _resolve_code_root()
_REPO = _CODE_ROOT.parent
for p in (_REPO, _REPO / "ablation_study" / "baseline_models"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common.datasets import BijieRawDataset
from model_architecture_cfm_landseg.data.datasets import BijieTripleStreamDataset, L4STripleStreamDataset
from model_architecture_cfm_landseg.tri_cfm_net import TriEncoderCFMNet


def load_checkpoint(
    ckpt_path: Path,
    *,
    ctx_ch: int,
    pyramid_width: int,
    flow_combine_scale: float,
    device: torch.device,
) -> Tuple[TriEncoderCFMNet, float]:
    state = torch.load(ckpt_path, map_location=device)
    model = TriEncoderCFMNet(
        rgb_ch=3,
        dem_ch=1,
        ctx_ch=ctx_ch,
        pyramid_width=pyramid_width,
        flow_combine_scale=flow_combine_scale,
        inference_flow_steps=0,
    )
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    thr = float(state.get("metric_threshold", 0.6))
    return model, thr


def _minmax_rgb_uint8(rgb_chw: np.ndarray) -> np.ndarray:
    x = rgb_chw.transpose(1, 2, 0).astype(np.float32)
    for c in range(3):
        mn, mx = float(x[:, :, c].min()), float(x[:, :, c].max())
        if mx > mn:
            x[:, :, c] = (x[:, :, c] - mn) / (mx - mn)
        x[:, :, c] = np.clip(x[:, :, c], 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def _mask_uint8(mask_hw: np.ndarray) -> np.ndarray:
    return ((mask_hw > 0).astype(np.uint8) * 255)


@torch.no_grad()
def predict_mask(
    model: TriEncoderCFMNet,
    batch: dict,
    device: torch.device,
    threshold: float,
) -> np.ndarray:
    rg = batch["stream_rgb"].float().to(device)
    dem = batch["stream_dem"].float().to(device)
    ctx = batch["stream_ctx"].float().to(device)
    out = model(rg, dem, ctx, gt_mask=None, inference_flow_steps=None)
    prob = torch.sigmoid(out["logits_aux"])[:, 0].cpu().numpy()
    return (prob >= threshold).astype(np.uint8)


def _pick_indices(n_total: int, n_pick: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    n_pick = min(n_pick, n_total)
    return sorted(rng.sample(range(n_total), n_pick))


def _bijie_stem(raw: BijieRawDataset, idx: int) -> str:
    return raw.files[idx].stem


def _export_bijie(
    *,
    out_dir: Path,
    raw: BijieRawDataset,
    indices: List[int],
    model: TriEncoderCFMNet,
    device: torch.device,
    threshold: float,
    resize_to: int,
    split_name: str,
) -> List[str]:
    triple = BijieTripleStreamDataset(raw, resize_to=resize_to, transform=None)
    subset = Subset(triple, indices)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems: List[str] = []
    for i in range(len(subset)):
        item = subset[i]
        stem = _bijie_stem(raw, indices[i])
        batch = {
            "stream_rgb": item["stream_rgb"].unsqueeze(0),
            "stream_dem": item["stream_dem"].unsqueeze(0),
            "stream_ctx": item["stream_ctx"].unsqueeze(0),
        }
        pred = predict_mask(model, batch, device, threshold)[0]
        rgb = item["stream_rgb"].numpy()
        gt = item["mask"].numpy()
        if gt.ndim == 3:
            gt = gt[0]
        Image.fromarray(_minmax_rgb_uint8(rgb)).save(out_dir / f"{stem}_image.png")
        Image.fromarray(_mask_uint8(gt)).save(out_dir / f"{stem}_mask_gt.png")
        Image.fromarray(_mask_uint8(pred)).save(out_dir / f"{stem}_mask_pred.png")
        stems.append(stem)
    (out_dir / "samples_list.txt").write_text(
        f"# {split_name}\n# threshold={threshold}\n" + "\n".join(stems) + "\n",
        encoding="utf-8",
    )
    return stems


def _export_l4s(
    *,
    out_dir: Path,
    root: Path,
    indices: List[int],
    model: TriEncoderCFMNet,
    device: torch.device,
    threshold: float,
    resize_to: int,
) -> List[str]:
    ds = L4STripleStreamDataset(root, ids=indices, resize_to=resize_to, transform=None)
    out_dir.mkdir(parents=True, exist_ok=True)
    stems: List[str] = []
    for i in range(len(ds)):
        item = ds[i]
        stem = ds.img_paths[i].stem.replace("image_", "")
        batch = {
            "stream_rgb": item["stream_rgb"].unsqueeze(0),
            "stream_dem": item["stream_dem"].float().unsqueeze(0),
            "stream_ctx": item["stream_ctx"].unsqueeze(0),
        }
        pred = predict_mask(model, batch, device, threshold)[0]
        rgb = item["stream_rgb"].numpy()
        gt = item["mask"].numpy()
        if gt.ndim == 3:
            gt = gt[0]
        Image.fromarray(_minmax_rgb_uint8(rgb)).save(out_dir / f"{stem}_image.png")
        Image.fromarray(_mask_uint8(gt)).save(out_dir / f"{stem}_mask_gt.png")
        Image.fromarray(_mask_uint8(pred)).save(out_dir / f"{stem}_mask_pred.png")
        stems.append(stem)
    (out_dir / "samples_list.txt").write_text(
        f"# landslide4sense\n# threshold={threshold}\n" + "\n".join(stems) + "\n",
        encoding="utf-8",
    )
    return stems


def _bijie_dataset_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "landslide").is_dir():
        return path
    inner = path / "Bijie-landslide-dataset"
    if inner.is_dir():
        return inner
    raise FileNotFoundError(f"Bijie layout not found under {path}")


def parse_args():
    p = argparse.ArgumentParser(description="Export qualitative segmentation triplets.")
    p.add_argument("--bijie-root", type=str, required=True)
    p.add_argument("--l4s-root", type=str, required=True)
    p.add_argument("--bijie-checkpoint", type=Path, default=None)
    p.add_argument("--l4s-checkpoint", type=Path, default=None)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_CODE_ROOT / "paper_submission_bundle/qualitative_segmentation_11",
    )
    p.add_argument("--num-samples", type=int, default=11)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resize-to", type=int, default=256)
    p.add_argument("--pyramid-width", type=int, default=64)
    p.add_argument("--flow-combine-scale", type=float, default=0.5)
    p.add_argument("--metric-threshold", type=float, default=-1.0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(
        args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )
    bijie_root = _bijie_dataset_root(Path(args.bijie_root))
    l4s_root = Path(args.l4s_root).expanduser().resolve()
    if not (l4s_root / "TrainData" / "img").is_dir():
        raise SystemExit(f"L4S TrainData/img not found under {l4s_root}")
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    bijie_ckpt = args.bijie_checkpoint or (_REPO / "runs/bijie/tri_encoder_cfm_v2/checkpoint/best.pt")
    l4s_ckpt = args.l4s_checkpoint or (
        _REPO / "runs/landslide4sense/tri_encoder_cfm_v2/checkpoint/best.pt"
    )
    if not bijie_ckpt.is_file():
        raise SystemExit(f"Bijie checkpoint not found: {bijie_ckpt}")
    if not l4s_ckpt.is_file():
        raise SystemExit(f"L4S checkpoint not found: {l4s_ckpt}")

    bijie_model, thr_b = load_checkpoint(
        bijie_ckpt,
        ctx_ch=4,
        pyramid_width=args.pyramid_width,
        flow_combine_scale=args.flow_combine_scale,
        device=device,
    )
    l4s_model, thr_l = load_checkpoint(
        l4s_ckpt,
        ctx_ch=6,
        pyramid_width=args.pyramid_width,
        flow_combine_scale=args.flow_combine_scale,
        device=device,
    )
    if args.metric_threshold >= 0:
        thr_b = thr_l = args.metric_threshold

    ls_raw = BijieRawDataset(bijie_root / "landslide", phase="landslide")
    nls_raw = BijieRawDataset(bijie_root / "non-landslide", phase="non-landslide")
    n_l4s = len(sorted((l4s_root / "TrainData" / "img").glob("*.h5")))

    ls_idx = _pick_indices(len(ls_raw), args.num_samples, args.seed)
    nls_idx = _pick_indices(len(nls_raw), args.num_samples, args.seed + 1)
    l4s_idx = _pick_indices(n_l4s, args.num_samples, args.seed + 2)

    dir_ls = out / "bijie_landslide"
    dir_nls = out / "bijie_non_landslide"
    dir_l4s = out / "landslide4sense"

    print(f"Bijie checkpoint: {bijie_ckpt} (threshold={thr_b})")
    print(f"L4S checkpoint: {l4s_ckpt} (threshold={thr_l})")
    print(f"Output: {out}")

    s1 = _export_bijie(
        out_dir=dir_ls,
        raw=ls_raw,
        indices=ls_idx,
        model=bijie_model,
        device=device,
        threshold=thr_b,
        resize_to=args.resize_to,
        split_name="bijie_landslide",
    )
    s2 = _export_bijie(
        out_dir=dir_nls,
        raw=nls_raw,
        indices=nls_idx,
        model=bijie_model,
        device=device,
        threshold=thr_b,
        resize_to=args.resize_to,
        split_name="bijie_non_landslide",
    )
    s3 = _export_l4s(
        out_dir=dir_l4s,
        root=l4s_root,
        indices=l4s_idx,
        model=l4s_model,
        device=device,
        threshold=thr_l,
        resize_to=args.resize_to,
    )

    (out / "README.txt").write_text(
        f"""Qualitative segmentation export ({args.num_samples} per folder)
================================================================

  bijie_landslide/      — Bijie landslide/ (GT + pred)
  bijie_non_landslide/  — Bijie non-landslide/ (empty GT; pred should be empty)
  landslide4sense/      — L4S TrainData (GT + pred)

Files per stem: {{stem}}_image.png, {{stem}}_mask_gt.png, {{stem}}_mask_pred.png

Checkpoints:
  Bijie: {bijie_ckpt} (thr={thr_b})
  L4S:   {l4s_ckpt} (thr={thr_l})

Stems:
  bijie_landslide: {', '.join(s1)}
  bijie_non_landslide: {', '.join(s2)}
  landslide4sense: {', '.join(s3)}
""",
        encoding="utf-8",
    )
    print("Done.")
    print(f"  {dir_ls.name}: {len(s1)}")
    print(f"  {dir_nls.name}: {len(s2)}")
    print(f"  {dir_l4s.name}: {len(s3)}")


if __name__ == "__main__":
    main()
