# Hybrid Landslide Architecture: Full Technical Specification

## 1) Model Intent and High-Level Design

This project implements a **two-stage hybrid system** for landslide mapping from remote-sensing imagery:

1. **Stage-1 binary classifier** filters non-landslide tiles quickly.
2. **Stage-2 GSAM segmenter** performs dense landslide delineation only on tiles likely to contain landslides.

This reduces unnecessary segmentation compute while preserving pixel-level quality on hard terrain boundaries.

---

## 2) Stage-1 Classifier (Implemented in `classfier_training.py`)

### 2.1 Input and preprocessing

- Input tile: `x in R^(3 x 128 x 128)` (RGB).
- Transform: resize to `128 x 128`, convert to tensor.

### 2.2 SE-enhanced convolutional blocks

Each `ConvBlock(c_in, c_out)` performs:

1. Convolution:
  `z = Conv3x3(x)` with padding 1.
2. Batch normalization:
  `z_hat = BN(z)`.
3. Activation:
  `a = ReLU(z_hat)`.
4. Channel recalibration (SE):
  - Squeeze: `s_c = (1/HW) * sum_{i,j} a_{c,i,j}`
  - Excitation: `e = sigma(W2 * ReLU(W1 * s))`
  - Rescale: `a'_c = e_c * a_c`

### 2.3 Classifier path

The implemented model (`ImprovedCNN`) is:

1. `Block1(3 -> 32)` + MaxPool2d(2): `32 x 64 x 64`
2. `Block2(32 -> 64)` + MaxPool2d(2): `64 x 32 x 32`
3. `Block3(64 -> 128)` + MaxPool2d(2): `128 x 16 x 16`
4. Conv4 `128 -> 256` + BN + ReLU + SE + MaxPool2d(2): `256 x 8 x 8`
5. Flatten: `256 * 8 * 8`
6. Dropout(0.5) -> FC(`16384 -> 512`) -> ReLU
7. Dropout(0.5) -> FC(`512 -> 2`) -> logits

Let logits be `l = [l0, l1]`, where class `1` is landslide.
Probability:

`p(landslide|x) = softmax(l)_1`

Decision rule:

- If `argmax(l) = 1`: send tile to segmentation stage.
- Else: reject tile from dense segmentation.

---

## 3) Stage-2 GSAM Segmentation Architecture

Stage-2 modifies SAM-B image encoding while keeping prompt and mask-decoder interfaces compatible.

### 3.1 SAM backbone loading and trainability policy (`builder.py`)

`build_gsam_vit_b(...)` does:

1. Load vanilla SAM-B checkpoint.
2. Replace `sam.image_encoder` with `GImageEncoder.from_pretrained(...)`.
3. Freeze all prompt-encoder parameters.
4. Freeze first `k` ViT blocks (`freeze_first_k`), keep later blocks trainable.
5. Optionally freeze mask decoder.

This creates a controlled fine-tuning regime:

- **Frozen**: prompt encoder + early transformer blocks.
- **Trainable**: PEG modules, CNN adapter branch, fusion 1x1 projection, later transformer blocks, and usually mask decoder.

---

## 4) GImageEncoder: ViT + PEG + CNN Fusion (`vit_wrapper.py`)

The custom image encoder extends `ImageEncoderViT`.

### 4.1 Token stream

Given image `I in R^(B x 3 x H x W)`:

1. Patch embedding creates token grid:
  `X0 in R^(B x h x w x C)`, where `h = H/16`, `w = W/16`.
2. For each transformer block index `i`:
  - Flatten to sequence `Si in R^(B x (h*w) x C)`.
  - Apply PEG:
  `Si' = PEG_i(Si, h, w)` to reinforce local positional structure.
  - Reshape back to grid.
  - If `i in fuse_blocks` (default `{4, 7, 10}`), fuse with CNN branch feature map.
  - Apply SAM ViT block `Block_i`.
3. Apply SAM neck to get decoder-ready embedding:
  `E in R^(B x 256 x h x w)`.

### 4.2 Fusion equation

At selected depth `i`, with token grid `Ti in R^(B x h x w x C)` and CNN features `F in R^(B x C x h x w)`:

1. Convert tokens to channels-first:
  `T_hat = permute(Ti) in R^(B x C x h x w)`.
2. Concatenate:
  `G = concat(T_hat, F)` -> `R^(B x 2C x h x w)`.
3. Project:
  `T_tilde = Conv1x1(G)` -> `R^(B x C x h x w)`.
4. Return to token-grid layout for ViT block input.

This updates global tokens with locality-heavy CNN cues.

---

## 5) CNN Adapter with Geology-Aware Attention (`cnn_adapter.py`)

CNN branch extracts high-frequency terrain structures at stride 16.

### 5.1 Backbone trunk

Default backbone is ConvNeXt-Base truncated at stride-16 output:

`F_raw in R^(B x C_out x h x w)`

Then:

`F_proj = Conv1x1(F_raw) in R^(B x C_embed x h x w)`

where `C_embed` matches ViT embed dimension.

### 5.2 Geology-aware gate

The `_GeoAttention` block computes:

1. Depthwise mask:
  `M = sigmoid(BN(DWConv3x3(F_proj)))`
2. Gated geology cue:
  `G_geo = F_proj * M`
3. Residual blend with learnable scalar `alpha`:
  `F_geo = F_proj + alpha * G_geo`

`alpha` is initialized at zero, so training starts near the original SAM behavior and gradually learns when terrain-specific enhancement is useful.

---

## 6) Decoder Interface and Losses (`train.py`)

### 6.1 Prompt handling

Training wrapper prepares:

- positional encoding from prompt encoder: `PE_32`
- dense no-mask embedding expanded to spatial grid: `D_prompt`
- zero sparse prompts

Then mask decoder is called with:

- `image_embeddings = E`
- `image_pe = PE_32`
- `sparse_prompt_embeddings = zeros(...)`
- `dense_prompt_embeddings = D_prompt`
- `multimask_output = False`

### 6.2 Output and resizing

Decoder logits are interpolated to `512 x 512`.

Let predicted logits be `Y_hat`, and ground truth mask be `Y`.

### 6.3 Objective

Total loss:

`L = BCEWithLogits(Y_hat, Y) + DiceLoss(Y_hat, Y)`

Dice component (as implemented):

1. `P = sigmoid(Y_hat)`
2. `Dice = (2 * sum(P*Y) + eps) / (sum(P) + sum(Y) + eps)`
3. `DiceLoss = 1 - mean(Dice)`

This balances per-pixel calibration (BCE) and region overlap quality (Dice).

---

## 7) Training Dynamics and Validation Protocol

Implemented training details in `train.py`:

- Optimizer: `AdamW(lr=1e-4, weight_decay=1e-2)`
- Epoch schedule: up to 40 epochs
- Early stopping on validation precision/recall trend
- Validation every 5 epochs
- Metrics: precision, recall, accuracy, ROC-AUC
- Artifacts: checkpoints, metrics CSV, ROC and PR plots

---

## 8) End-to-End Operational Pipeline

1. Read remote-sensing tile.
2. Classifier predicts landslide/non-landslide.
3. If non-landslide: stop here (fast rejection).
4. If landslide:
  - pass full-resolution tile to GSAM,
  - encode with ViT + PEG + CNNAdapter + geology gate,
  - decode mask through SAM mask decoder,
  - upsample and threshold/post-process as needed.
5. Aggregate tile-level masks into larger geographic map.

---

## 9) Why This Hybrid Design Works

### 9.1 Efficiency gain

Classifier gate avoids running expensive segmentation on clear negatives.

### 9.2 Representation gain

SAM supplies strong global priors; CNNAdapter restores local texture and boundary sensitivity.

### 9.3 Domain adaptation gain

Geology-aware gating introduces terrain-specific modulation without requiring explicit geology labels.

### 9.4 Stability gain

Selective freezing preserves pretrained generalization while enabling targeted adaptation to landslide morphology.

---

## 10) Implementation Consistency Notes

- `SimpleCNN` in `classifier.py` had a fully connected input-size mismatch; for `128x128` input after three pool layers, spatial size is `16x16`, not `32x32`.
- This has been aligned so the flattened feature size matches the actual tensor shape.
- The main trained classifier in `classfier_training.py` already uses the deeper `ImprovedCNN` variant and remains the primary stage-1 model.

---

## 11) Suggested Figure Breakdown (for paper and slides)

Use a four-panel hierarchy:

1. **Panel A**: end-to-end gated two-stage pipeline.
2. **Panel B**: classifier internals (Conv-SE stack and logits).
3. **Panel C**: GSAM encoder internals (PEG + CNN fusion at selected depths).
4. **Panel D**: decoder/loss/metrics loop.

This decomposition matches both reproducibility and reviewer readability.