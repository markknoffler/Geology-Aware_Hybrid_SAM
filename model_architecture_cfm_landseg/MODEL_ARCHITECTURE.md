# TriEncoderCFMNet: triple-stream encoders, gated multi-scale fusion, and conditional flow matching

This document specifies the **TriEncoderCFMNet** architecture implemented in this package so that a reader can re-implement the forward pass, losses, and Euler inference without reading the PyTorch source line-by-line. File references are relative to `[SAM/model_architecture_cfm_landseg/](.)`

## 1. Problem and notation

- Input example: H \times W landslide image(s) with topography. The model consumes **three tensors** at full resolution:
  - **RGB stream** \mathbf{X}^{rgb} \in \mathbb{R}^{B \times 3 \times H \times W} (multispectral composites are allowed in code as long as C=3 for this stem).
  - **DEM stream** \mathbf{X}^{dem} \in \mathbb{R}^{B \times 1 \times H \times W} (single band elevation, min–max normalized per sample as in baseline datasets).
  - **Context stream** \mathbf{X}^{ctx} \in \mathbb{R}^{B \times C_{ctx} \times H \times W} with C_{ctx}=6 on Landslide4Sense (concatenation of RGB and [NDVI, slope, DEM] spectral stack) and C_{ctx}=4 on Bijie (RGB \Vert DEM).
- Target binary mask \mathbf{Y} \in 0,1^{B \times 1 \times H \times W}.
- Output **segmentation logits** \hat{\mathbf{L}} \in \mathbb{R}^{B \times 1 \times H \times W} after optional FM refinement (see §6).

## 2. High-level data flow

```
X_rgb ──► PyramidEncoder_ecnn ──┐
                                  ├──► TrimodalGatedFusion (per scale) ──► fused pyramid {F^0,F^1,F^2}
X_dem ──► PyramidEncoder_edem ──┤         │
        (DEM + |∇H| pre-stem)    │         ├──► Aux segmentation head ──► logits_aux (full res)
                                  │         │
X_ctx ──► PyramidEncoder_ecntx ───┘         └──► Velocity U-Net v_θ(x_t, t | F) ──► FM training / Euler inference
```

## 3. Triple-stream encoders (pyramid)

Each stream uses the same `**PyramidEncoder**` layout (see `[encoders/cnn_pyramid.py](encoders/cnn_pyramid.py)`).

### 3.1 Shared operator definitions

- **Conv–BN–ReLU**  \psi_{c_{out}}(\mathbf{x}) = \mathrm{ReLU}( \mathrm{BN}( \mathrm{Conv}*{7/4}(\mathbf{x}))) for the stem, where \mathrm{Conv}*{7/4} denotes kernel 7, stride 4 (spatial downsample \times 4).
- **Residual block**  \mathcal{R}_c(\mathbf{x}) = \mathrm{ReLU}( \mathbf{x} + \mathcal{F}_c(\mathbf{x})) where \mathcal{F}_c is two 3\times3 conv–BN–ReLU–conv–BN stacks in c channels.
- **Down block** \mathcal{D}*{c \to 2c}(\mathbf{x}) = \mathrm{ReLU}(\mathrm{BN}(\mathrm{Conv}*{3/2}(\mathbf{x}))) (stride 2).

Let base width w (CLI `--pyramid_width`, default 48).

### 3.2 Stage channel schedule

For any stream, after stem and two down steps the feature maps are:


| Level \ell | Spatial size (for H=W=256)        | Channels C_\ell |
| ---------- | --------------------------------- | --------------- |
| \ell=0     | \tfrac{H}{4} \times \tfrac{W}{4}  | w               |
| \ell=1     | \tfrac{H}{8} \times \tfrac{W}{8}  | 2w              |
| \ell=2     | \tfrac{H}{16}\times \tfrac{W}{16} | 3w              |


Per stream output:  \mathbf{A}^{(\ell)}, \mathbf{B}^{(\ell)}, \mathbf{C}^{(\ell)} _{\ell=0}^2 for RGB, DEM, and context.

### 3.3 DEM slope prior (encoder branch)

Before the DEM stem, compute finite-difference gradients (differentiable):


(\nabla_x H)*{i,j} = H*{i,j+1} - H_{i,j}, \quad (\nabla_y H)*{i+1,j} = H*{i+1,j} - H_{i,j}


with replicate padding. Slope magnitude \nabla H = \sqrt{(\nabla_x H)^2 + (\nabla_y H)^2 + \varepsilon}.

The DEM encoder input is \mathbf{X}^{dem\prime} = [\mathbf{X}^{dem} \Vert \nabla H ] \in \mathbb{R}^{B \times 2 \times H \times W} followed by the usual stem when `dem_branch=True`.

**Rationale**: Encourages the topography branch to emphasize geomorphic relief that correlates with slope-driven failure, and matches the geomorph loss (§7.2).

## 4. Multi-scale trimodal gated fusion

At each level \ell, tensors \mathbf{A}^{(\ell)}, \mathbf{B}^{(\ell)}, \mathbf{C}^{(\ell)} \in \mathbb{R}^{B \times C_\ell \times h_\ell \times w_\ell} share identical shapes.

### 4.1 Global stream weights (softmax gate)

Concatenate \mathbf{Z}^{(\ell)} = [\mathbf{A}^{(\ell)} \Vert \mathbf{B}^{(\ell)} \Vert \mathbf{C}^{(\ell)}] along channel (3C_\ell channels).

GAP: \mathbf{g}=\mathrm{GAP}(\mathbf{Z}^{(\ell)}) \in \mathbb{R}^{B \times 3C_\ell \times 1 \times 1}.

Two-layer 1\times1 convolutional MLP yields \boldsymbol{\ell}^{(\ell)} \in \mathbb{R}^{B \times 3 \times 1 \times 1}.

Stream weights:


(w_0, w_1, w_2) = \mathrm{Softmax}(\boldsymbol{\ell}^{(\ell)}, \text{dim}=1), \quad \sum_i w_i = 1 .


Global mixture:

\mathbf{M}^{(\ell)} = w_0 \odot \mathbf{A}^{(\ell)} + w_1 \odot \mathbf{B}^{(\ell)} + w_2 \odot \mathbf{C}^{(\ell)}

(broadcasting w_i over channels).

### 4.2 Spatial modulation map \gamma^{(\ell)}

A depthwise-heavy path (7×7 grouped conv) maps \mathbf{Z}^{(\ell)} \to \gamma^{(\ell)} \in [0,1]^{B \times C_\ell \times h_\ell \times w_\ell} with sigmoid.

Fusion output:


\mathbf{F}^{(\ell)} = \mathrm{BN}\left( \gamma^{(\ell)} \odot \mathbf{M}^{(\ell)} + (1-\gamma^{(\ell)}) \odot \tfrac{1}{3}(\mathbf{A}^{(\ell)}+\mathbf{B}^{(\ell)}+\mathbf{C}^{(\ell)}) \right).


**Interpretation**: Global gates pick which modality dominates each **example**; \gamma reallocates confidence **spatially**, similar in spirit to PMCNet-style dynamic fusion but without full self-attention.

## 5. Auxiliary segmentation head

The finest fused map \mathbf{F}^{(0)} (stride-4) feeds a two-layer conv header:


\hat{\mathbf{L}}*{aux} = \mathrm{Conv}*{1\times1}(\mathrm{ReLU}(\mathrm{BN}(\mathrm{Conv}_{3\times3}(\mathbf{F}^{(0)}))))


Upsample bilinearly to (H,W). This path provides **stable discriminative gradients** and is the default training-time source of \hat{\mathbf{L}} in the loss unless FM integration is enabled at validation.

## 6. Conditional flow matching (training + inference)

### 6.1 Latent target  \mathbf{z}

From ground-truth mask \mathbf{Y} (values in 0,1), form probability map with clipping \epsilon_m:


p = \mathrm{clip}(\mathbf{Y}, \epsilon_m, 1-\epsilon_m), \quad \mathbf{z} = \sigma \cdot \mathrm{logit}(p) = \sigma \cdot \log\frac{p}{1-p}

with scale \sigma = `--latent_sigma` (default 4). This maps binary labels into a smooth latent suitable for regression.

### 6.2 Straight-path interpolation and target velocity

Sample t \sim \mathcal{U}[0,1] per batch element, and \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) matching shape of \mathbf{z}.

Interpolation (OT straight path between data and noise):


\mathbf{x}_t = (1-t)\mathbf{z} + t\boldsymbol{\epsilon}, \qquad \mathbf{x}_t \in \mathbb{R}^{B\times 1\times H\times W}.


The **conditional velocity field** satisfies (for this path):


\mathbf{v}^{\star}(\mathbf{x}_t, t) = \boldsymbol{\epsilon} - \mathbf{z}

(independent of \mathbf{x}_t along the exact straight line; the network learns the **conditional** extension off the line).

The neural velocity \mathbf{v}_\theta(\mathbf{x}_t, t \mid \mathbf{F}) is parameterized by `**VelocityConditionalUNet`** (hourglass; see `[decoders/velocity_unet.py](decoders/velocity_unet.py)`).

**FM loss**:

\mathcal{L}*{fm} = \mathbb{E}*{t,\boldsymbol{\epsilon}}\left[ \mathbf{v}_\theta - (\boldsymbol{\epsilon}-\mathbf{z})_2^2 \right].


### 6.3 Time smoothness regularizer (finite-difference surrogate)

With the **same** \mathbf{x}_t, sample t' = \min(t+\Delta, 1) (\Delta{=}0.1 in code). Penalize sensitivity of velocity to time:


\mathcal{L}*{v} = \mathbb{E}\left[ \mathbf{v}*\theta(\mathbf{x}*t, t) - \mathbf{v}*\theta(\mathbf{x}_t, t')_2^2 \right].


This loosely approximates a \partial_t \mathbf{v}_\theta^2 regularizer (least-action style stability).

### 6.4 Architecture of the velocity hourglass (summary)

- Time embedding: MLP on scalar t → \mathbf{e}_t \in \mathbb{R}^{d_t}.
- Latent stem \mathbf{x}_t: 7\times7, stride 4 → match \mathbf{F}^{(0)} resolution; concat with \mathbf{F}^{(0)} then 1\times1 blend.
- **FiLM-style blocks** (`FiLM2d`): group norm on features, then \gamma,\beta predicted from [\mathrm{GAP}(\mathbf{F}^{(\ell)})\Vert \mathbf{e}_t].
- Down–up U with skip joins; output 1\times1 conv yields \mathbf{v}_\theta upsampled to full (H,W).

### 6.5 Euler integration at inference

At validation (if `--val_infer_flow_steps` / `model_flow_steps` K>0):

1. Sample \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0},\mathbf{I}) (deterministic per forward pass).
2. Initialize \mathbf{x} \leftarrow \boldsymbol{\epsilon}.
3. For k=0,\ldots,K-1, let t_k = \max(1-k/K, 10^{-3}), step size \Delta t = 1/K:

   \mathbf{x} \leftarrow \mathbf{x} - \Delta t  \mathbf{v}_\theta(\mathbf{x}, t_k \mid \mathbf{F})
   

Readout logits from final latent: \hat{\mathbf{L}}*{flow} = \mathrm{Conv}*{1\times1}^{ro}(\mathbf{x}).

**Combined output** (evaluation):

\hat{\mathbf{L}} = \hat{\mathbf{L}}*{aux} + \alpha*{cfm}  \hat{\mathbf{L}}*{flow}

with \alpha*{cfm} = `--flow_combine_scale` (default 0.5).

**Training** does not run Euler by default (faster); only \mathcal{L}_{fm} trains the velocity field.

## 7. Loss functions

### 7.0 Why `train_loss` and `val_loss` can look unrelated (before normalization)

- **Training** runs the model with `gt_mask` set, so the loss includes **conditional flow matching** \mathcal{L}*{fm}=v*\theta-(\boldsymbol\epsilon-\mathbf{z})^2 and **time-smoothness** terms. Target \boldsymbol\epsilon-\mathbf{z} uses \mathbf{z}=\sigma\cdot\mathrm{logit}(\mathbf{y}): for binary \mathbf{y}, |\mathbf{z}| is often \mathcal{O}(10\text{–}30). A random velocity network therefore contributes a **large raw MSE** (often \sim10^2 per-pixel mean in early epochs)—this is numerical scale, not “wrong HDF5 reads.”
- **Validation** forward uses `gt_mask=None`, so **FM terms are not computed** at all; `**val_loss` is essentially Tversky + geomorph only** (similar order to \sim 1\text{–}5).
- Implementations divide \mathcal{L}*{fm} and the time-smooth term by `**fm_residual_scale_sq`** (CLI `--fm_residual_scale_sq`, default ( \sigma\cdot 6 )^2) so **logged `train_loss` is comparable in magnitude** to `val_loss` while preserving relative gradient contributions to v*\theta.

Total training objective (weighted sum, see `[losses/composite.py](losses/composite.py)`):


\mathcal{L} = \lambda_{seg}\mathcal{L}*{Tversky}(\hat{\mathbf{L}}*{aux}, \mathbf{Y})

- \lambda_{fm}\mathcal{L}_{fm}
- \lambda_{v}\mathcal{L}_{v}
- \lambda_{geo}\mathcal{L}_{geo}.


Defaults: \lambda_{seg}{=}2,\ \lambda_{fm}{=}1,\ \lambda_{v}{=}0.05,\ \lambda_{geo}{=}0.15 (all CLI-tunable).

### 7.1 Tversky loss (segmentation)

With probabilities p=\sigma(\hat{\mathbf{L}}_{aux}), flattened over pixels:


TV = \frac{TP + \epsilon}{TP + \alpha FP + \beta FN + \epsilon}, \quad \mathcal{L}_{Tversky} = 1 - TV.


Matches baseline choices (`--tversky_alpha`, `--tversky_beta` aligned with `[common/losses.py](../ablation_study/baseline_models/common/losses.py)`).

### 7.2 Geomorphological alignment \mathcal{L}_{geo}

Let m=\sigma(\hat{\mathbf{L}}_{aux}). Gradients (\partial_x m, \partial_y m) and (\partial_x H, \partial_y H) via finite differences (replicate padding).

Slope energy S = \sqrt{(\partial_x H)^2+(\partial_y H)^2}. Normalize by batch mean for scale invariance:


w = \frac{1}{1 + 5 S / (\bar{S}+ \epsilon)} , \quad 
\mathcal{L}_{geo} = \mathbb{E}\left[ ((\partial_x m)^2 + (\partial_y m)^2) \cdot w \right].


High weight on flat terrain suppresses spurious mask boundary energy; steep areas allow sharper boundaries.

## 8. Relation to dual-stream DiGATe baselines

The repository’s **dual-stream gated** reference implementation lives at `[SAM/ablation_study/dual_stream_gated/](../ablation_study/dual_stream_gated/)` (two EfficientNet towers with gated fusion before a UNet decoder). **TriEncoderCFMNet** differs as follows:


| Aspect  | dual_stream_gated           | TriEncoderCFMNet                                        |
| ------- | --------------------------- | ------------------------------------------------------- |
| Streams | 2 (RGB vs NDVI–slope–DEM)   | 3 (RGB, DEM/slope, RGB+modal context)                   |
| Fusion  | Gates between two towers    | Trimodal softmax + spatial \gamma **per pyramid level** |
| Decoder | Classical UNet segmentation | Aux UNet-style head + **conditional FM velocity UNet**  |
| Loss    | Tversky (+ aux in model)    | Tversky + FM + time-smooth + topography prior           |


**Paper PDF**: If you add `[SAM/literature_review/dual_stream.pdf](../literature_review/dual_stream.pdf)`, cite it in your manuscript for the dual-stream inductive bias; the training code here intentionally uses **baseline splits** from `[common/datasets.py::build_l4s_split](../ablation_study/baseline_models/common/datasets.py)` and `build_bijie_split`, **not** alternate paper-only splits.

## 9. Training I/O contract (baseline parity)

Trainer in `[training/train.py](training/train.py)` mirrors `[common/trainer.py::train_model](../ablation_study/baseline_models/common/trainer.py)`:

- Layout: `{output_dir}/{dataset}/{experiment_name}/{checkpoint/,results/}`.
- Epoch CSV `[results/epoch_metrics.csv](training/train.py)` columns: train/val `loss`, pixel `acc,precision,recall,f1,iou`, image `auroc,auprc,image_best_f1,image_best_threshold`.
- Checkpoints `checkpoint/epoch_XXXX.pt`, `checkpoint/best.pt`; resume restores `optimizer` + `epoch` + best F1 tracker.

## 10. How to run

From the `SAM/` directory:

```bash
python -m model_architecture_cfm_landseg.training.train \
  --dataset landslide4sense \
  --dataset_root /path/to/Landslide4Sense \
  --output_dir ./runs \
  --experiment_name tri_encoder_cfm \
  --epochs 200
```

Optional FM at validation:

```bash
... --model_flow_steps 6 --val_infer_flow_steps 6
```

## 11. Optional extensions (not in v1 code)

- **Mamba branch**: Swap `PyramidEncoder_ecntx` selective scan (requires optional `mamba-ssm`).
- **KAN fusion spline mixer**: Higher-order nonlinear fusion atop \mathbf{Z}^{(\ell)} (heavy tuning workload).
- **RK45 ODE solver** for inference instead of Euler (better trajectory accuracy, higher cost).

