# Baseline Models

This directory provides baseline training pipelines for:

- `unet`
- `dual_stream_unet`
- `linknet`
- `deeplabv3plus`
- `transunet`
- `shapeformer`
- `rmau_net`
- `dep_unet`
- `emr_hrnet`
- `gmnet`

Each model folder contains:

- `model.py`
- `losses.py`
- `training.py`

Shared components are implemented in `common/`:

- dataset handling for Landslide4Sense (`.h5`) and Bijie (`.png`)
- paper-style Tversky loss settings
- training/evaluation loops with metrics: Acc, Precision, Recall, F1, IoU (plus image-level AUROC/AUPRC)
- checkpointing every 5 epochs
- resume from latest checkpoint (`--resume`)
- metric logging to `results/epoch_metrics.csv` and `results/final_metrics.csv`

## Dataset behavior

- **Landslide4Sense**: uses `TrainData` only and creates a 90/10 train/validation split from training data.
- **Bijie**: uses official class-wise 70/20/10 split from `landslide` and `non-landslide`.
- **Baseline input policy (paper-aligned)**:
  - Where the paper explicitly reports Landslide4Sense bands, defaults match those settings (e.g., `unet: rgb_swir`, `deeplabv3plus: ngb`).
  - For models without explicit per-model band specification in the paper text, defaults remain optical single-stream unless overridden by CLI flags.
  - `dual_stream_unet` remains multimodal.

## Example commands

UNet on Landslide4Sense:

```bash
python codebase/ablation_study/baseline_models/unet/training.py \
  --dataset landslide4sense \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset \
  --output_dir codebase/ablation_study/baseline_models
```

Dual-stream UNet on Bijie:

```bash
python codebase/ablation_study/baseline_models/dual_stream_unet/training.py \
  --dataset bijie \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide/Bijie-landslide-dataset \
  --output_dir codebase/ablation_study/baseline_models
```

Resume training:

```bash
python codebase/ablation_study/baseline_models/gmnet/training.py \
  --dataset bijie \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide/Bijie-landslide-dataset \
  --output_dir codebase/ablation_study/baseline_models \
  --resume
```
