# Dual-Stream Gated Landslide Segmentation

This directory now mirrors the released DiGATe-UNet codepath from the paper as closely as possible within the local flat-file layout.

## What Matches the Paper

- Siamese dual-stream encoder-decoder architecture.
- `EfficientNet-B4` backbone by default via `timm`.
- Pretrained `EfficientNet-B4` with `freeze_backbone=True` by default.
- `TransUp`, `UpFlex`, `GateFuse`, and deep supervision (`main`, `aux2`, `aux3`).
- Landslide4Sense setup using `RGB` and `NDVI + Slope + DEM`.
- Resize to `256x256`.
- Training augmentations: horizontal flip, vertical flip, Gaussian noise, salt-and-pepper noise, and CLAHE.
- Tversky loss with `alpha=0.3`, `beta=0.7`, plus gate regularization.

## Dataset Layout Expected

```
dataset/
  TrainData/
    img/image_*.h5
    mask/mask_*.h5
  ValidData/
    img/image_*.h5
    mask/mask_*.h5   # optional
  TestData/
    img/image_*.h5
```

## Train

```bash
python training.py \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset \
  --output_dir . \
  --backbone tf_efficientnet_b4 \
  --epochs 100 \
  --batch_size 32 \
  --save_every 5
```

## Resume from Last Checkpoint

```bash
python training.py \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset \
  --output_dir . \
  --resume
```

## Outputs

- `checkpoint/epoch_XXXX.pt` saved every 5 epochs.
- `checkpoint/best.pt` best validation F1.
- `results/epoch_metrics.csv` epoch-wise training/validation metrics.
- `results/final_metrics.csv` final run summary.

## Optional Dataset Check

```bash
python data_processing.py \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset \
  --output_json ./results/dataset_summary.json
```

## Notes

- Main defaults:
  - `--backbone tf_efficientnet_b4`
  - `--resize_to 256`
  - `--bands RGB-NDVI-SLOPE-DEM`
  - `--pretrained`
  - `--freeze_backbone`
  - separate encoders by default to match the released notebook path; add `--share_backbone` to force siamese sharing
- Current tuning defaults are slightly more precision-friendly than the paper:
  - `--tversky_alpha 0.6`
  - `--tversky_beta 0.4`
  - `--metric_threshold 0.6`
- To reproduce the paper settings exactly, use:
  - `--tversky_alpha 0.3 --tversky_beta 0.7 --metric_threshold 0.5`
- Bijie training:

  Create a new run using `train_bijie.py` (PNG loader for Bijie + paper split 70/20/10).

  Example:

```bash
python train_bijie.py \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide/Bijie-landslide-dataset \
  --output_dir ./outputs_bijie \
  --epochs 100 \
  --batch_size 32 \
  --lr 3e-4 \
  --weight_decay 1e-4 \
  --backbone tf_efficientnet_b4 \
  --pretrained \
  --freeze_backbone \
  --share_backbone \
  --tversky_alpha 0.3 \
  --tversky_beta 0.7 \
  --main_weight 1.0 \
  --aux2_weight 0.6 \
  --aux3_weight 0.4 \
  --reg_weight 1e-3 \
  --metric_threshold 0.5
```

- Note: `train_bijie.py` always uses `resize_to=256` and uses `RGB` + `DEM replicated to 3 channels` (paper setting).
- Required runtime packages now include `timm` and `segmentation-models-pytorch`.
