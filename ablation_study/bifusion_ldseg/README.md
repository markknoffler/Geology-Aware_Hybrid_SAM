# BiFusion-LDSeg (Ablation Study on Landslide4Sense)

This directory contains a Landslide4Sense adaptation of the `BiFusion-LDSeg` paper-style pipeline:
- Bi-directional CNN/Transformer fusion (Bi-AG)
- Latent diffusion-style denoising
- Boundary-aware progressive decoder with reverse attention

## Important Setup Choice

- The original BiFusion-LDSeg paper uses optical imagery.
- This implementation also defaults to optical-only input on Landslide4Sense (RGB only).
- DEM/Slope channels are intentionally not used in this ablation setup.

## Files

- `dataset.py`: TrainData H5 loader (image/mask paired, RGB-only).
- `model.py`: BiFusion-LDSeg model.
- `losses.py`: Composite training objective (BCE + Dice + diffusion + deep supervision).
- `metrics.py`: IoU, DSC, ASSD, HD, ECE, Precision, Recall, F1, Accuracy.
- `training.py`: End-to-end train/validate/test script with CSV logging and checkpointing.
- `data_processing.py`: Optional dataset shape/channel summary helper.

## Dataset

Expected root:

```
/home/user/Desktop/Deep_learning_projects/4PI/dataset
  TrainData/
    img/image_*.h5
    mask/mask_*.h5
```

Training uses **only `TrainData`** and performs deterministic split:
- Train: 80%
- Validation: 10%
- Test: 10%

## Train + Validate + Test

```bash
python training.py \
  --dataset_root /home/user/Desktop/Deep_learning_projects/4PI/dataset \
  --output_dir . \
  --epochs 100 \
  --batch_size 16 \
  --save_every 10
```

## Outputs

- `checkpoints/epoch_XXXX.pt` every 10 epochs
- `checkpoints/best.pt` best model by validation DSC
- `results/epoch_metrics.csv` per-epoch train/val metrics
- `results/final_metrics.csv` final test metrics on held-out 10% split
