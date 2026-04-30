# DEM-Native Geomamba-SAM Architecture

## Overview

`geomamba_sam` is a single-stage landslide segmentation branch that treats DEM as a first-class modality inside the encoder stack.
Unlike the previous two-stage system, this branch does not run a separate classifier gate before segmentation.
The model learns to emit near-zero masks for non-landslide scenes through direct binary mask supervision.

## Core Design

Implemented in `models/geomamba_sam.py`.

### 1) Dual token streams

For each sample:

- RGB image: `I_rgb in R^(B x 3 x H x W)`
- DEM map: `I_dem in R^(B x 1 x H x W)`

Patch embedding yields:

- `X_rgb^0 in R^(B x h x w x C)`
- `X_dem^0 in R^(B x h x w x C)`

where `h = H / p`, `w = W / p`, `p` is patch size from SAM-ViT.

### 2) Per-depth multimodal fusion

For each block depth `l`:

1. Positional refresh:
   - `X_rgb <- PEG_rgb^l(X_rgb)`
   - `X_dem <- PEG_dem^l(X_dem)`
2. Cross-modal token attention:
   - RGB queries DEM tokens.
   - DEM queries RGB tokens.
3. Tri-gate modulation:
   - channel gate `G_c` from global pooled descriptors.
   - spatial gate `G_s` from local fused maps.
   - confidence gate `G_u` from DEM variability proxy.
   - final gate: `G = G_c * G_s * G_u`
4. Gated update:
   - `X_rgb <- X_rgb + G * C_rgb`
   - `X_dem <- X_dem + (1-G) * C_dem`
5. GeoState block (state-style terrain propagation) on RGB stream.
6. SAM ViT block on updated RGB stream.

Final fusion before neck:

- `X_rgb <- X_rgb + W_dem(X_dem)`

Neck projection:

- `E in R^(B x 256 x h x w)`

### 3) Decoder coupling

Decoder remains SAM-compatible:

- empty sparse prompts
- dense no-mask prompt embedding
- prompt positional encoding
- `multimask_output=False`

Output logits are resized to target mask resolution.

## Objective

Per batch, imbalance-aware objective:

`L = L_tversky`

with:

- `TP = sum(P * M)`
- `FP = sum(P * (1 - M))`
- `FN = sum((1 - P) * M)`
- `TI = (TP + eps) / (TP + alpha*FP + beta*FN + eps)`
- `L_tversky = 1 - mean(TI)`
- `P = sigmoid(M_hat)`

Default hyperparameters in training script:

- `alpha = 0.7`
- `beta = 0.3`

## Dataset Protocols

## Landslide4Sense (`.h5`)

Expected structure:

- `TrainData/img/*.h5`
- `TrainData/mask/*.h5`
- `ValidData/img/*.h5` (optional unlabeled)
- `TestData/img/*.h5` (optional unlabeled)

Training policy:

- uses only `TrainData` labeled samples
- deterministic `90/10` split from `TrainData` into train/test
- test-10% is never used for training
- test metrics are computed each epoch

Key detection (configurable fallback):

- image keys: `img|image|images|x|X`
- mask keys: `mask|masks|label|labels|y|Y`

Channel convention:

- RGB defaults to channels `(0,1,2)`
- DEM defaults to channel `3`

## Bijie

Expected structure:

- `Bijie-landslide-dataset/landslide/image/*.png`
- `Bijie-landslide-dataset/landslide/dem/*.png`
- `Bijie-landslide-dataset/landslide/mask/*.png`
- `Bijie-landslide-dataset/non-landslide/image/*.png`
- `Bijie-landslide-dataset/non-landslide/dem/*.png`

Mask behavior:

- landslide samples use dataset masks
- non-landslide samples synthesize all-black masks

Split behavior:

- if split files exist (`train.txt`, `val.txt`, `test.txt`), they are used directly
- otherwise deterministic stratified `70/20/10` is applied

## Training Pipeline

Implemented in `train.py` with required flags:

- `--dataset {landslide4sense,bijie}`
- `--dataset-root`
- `--sam-checkpoint`
- `--results-dir`

Additional controls:

- `--epochs`, `--batch-size`, `--num-workers`
- `--lr`, `--weight-decay`
- `--freeze-first-k`
- `--target-size`, `--seed`
- `--save-every`
- `--resume {auto|none|/path/to/ckpt.pt}`

## Checkpoint/Resume

- Checkpoints saved every `save-every` epochs (default 5) in:
  - `results/<dataset>/checkpoints/epoch_XXX.pt`
- Resume modes:
  - `auto` -> latest checkpoint in checkpoint directory
  - explicit path -> that checkpoint
  - `none` -> fresh training

## Metrics and Logging

Logged each epoch for test set (and val for Bijie):

- `accuracy`
- `precision`
- `recall`
- `f1`
- `iou`
- `dice`
- `auroc`
- `auprc`
- `best_f1`
- `best_threshold`

System-table placeholders (kept for paper compatibility, filled as available):

- `fps`
- `peak_memory_mb`
- `gflops`
- `trainable_params_m`

Saved at:

- `results/<dataset>/metrics/metrics.csv`

## Why this branch is structurally different

The previous branch fused auxiliary features at limited points.
This branch embeds DEM at the token backbone level through repeated cross-modal attention + gating + terrain-state propagation, making DEM a native part of representation learning across depth.
