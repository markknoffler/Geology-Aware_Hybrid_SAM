# Image-level landslide vs non-landslide evaluation (TriEncoderCFMNet)

This document explains **why** we run a separate presence-detection evaluation, **what** it measures, **how** the code implements it, and **where** to paste results into the manuscript after you execute the script on your training server.

It complements the pixel-level ablation tables in `bijie_best_validation_summary.csv` and `landslide4sense_best_validation_summary.csv`, which answer “how well are landslide pixels segmented?” Your mentor’s question is different: **does the model stay silent on scenes with no landslide and activate on scenes that contain landslide?** For a segmentation network that means: **predict an empty (black) mask when GT is empty, and predict foreground when GT contains landslide.**

---

## 1. Why this evaluation exists

| Concern | Pixel F1 / IoU on val split | Presence evaluation (this task) |
| --- | --- | --- |
| Question | How accurate is the mask boundary? | Did the model **declare** a landslide on this **image**? |
| Bijie non-landslide folder | Often diluted in mixed metrics | Explicit **GT = empty mask** tiles |
| False alarms | Hidden in high accuracy | **False alarm rate** on non-landslide images |
| Missed events | Partially reflected in recall | **Sensitivity** on landslide images |

Segmentation and presence are linked but not identical: a model can achieve moderate pixel F1 while still firing spurious blobs on negative tiles (hurting trust in deployment), or achieve high pixel scores on positives while missing entire negative-set discipline.

---

## 2. What we measure

### 2.1 Ground-truth image label

For every validation image after the same resize/preprocessing as training:

- **GT landslide (`gt_present = 1`)**: any foreground in the binary mask (`mask > 0` after resize).
- **GT non-landslide (`gt_present = 0`)**: mask is entirely background.

**Bijie:** validation concat uses the official 70/20/10 stratified split from `build_bijie_split()` — landslide PNGs with masks **plus** `non-landslide/image` tiles with **implicit all-zero masks** (no mask files).

**Landslide4Sense:** validation is 10% of `TrainData/img` indices (`build_l4s_split`, seed 42). There is **no separate negative-image folder**; “non-landslide” here means **competition tiles whose mask is empty**. Count may be small; the script still reports `n_images` so you can state limitations honestly in the paper.

### 2.2 Predicted image label (primary rule)

Matches the **image-level metric hook already used in training** (`image_level_metrics_from_logits` in `SAM/ablation_study/baseline_models/common/metrics.py`):

1. Binarize sigmoid probabilities at `prob_threshold` (default **0.6**, read from checkpoint unless `--metric-threshold` is set).
2. Extract connected components with area ≥ `min_connected_area_px` (default **20**).
3. **Predicted positive** (`pred_present_instance = 1`) if **≥ 1** component survives.
4. **Image score** (for histogram / ROC): maximum probability inside the largest qualifying component, or **0** if none.

Auxiliary columns in per-image CSVs:

- `pred_present_area`: ≥ `min_area` foreground pixels (stricter “black mask” view).
- `pred_fg_fraction`: fraction of pixels above threshold.
- `max_prob`: global max probability (diagnostic).

### 2.3 Aggregated metrics (combined table)

For each **dataset** × **GT class** row:

| GT class | Primary rate column | Meaning |
| --- | --- | --- |
| `landslide` | `detection_rate` (= sensitivity) | Fraction of landslide images with `pred_present_instance = 1` |
| `non_landslide` | `sensitivity_or_specificity` (= specificity) | Fraction of negative images with **no** false activation |
| `non_landslide` | `false_alarm_rate` | Fraction of negative images with a false activation |
| `all_images` | `image_auroc`, `image_auprc`, `image_best_f1` | Pooled ranking by `image_score` (both classes) |

Pixel columns (`mean_pixel_f1`, `mean_pixel_iou`, …) are **within-group means** so you can show that negatives also have near-zero overlap when the model behaves.

---

## 3. Which checkpoint and epoch

The script **does not** re-search epochs. It reads **`best_epoch`** for `tri_encoder_cfm_v2` from:

- `SAM/resources/results/bijie_ablation_report/bijie_best_validation_summary.csv` → currently **112**
- `SAM/resources/results/l4s_ablation_report/landslide4sense_best_validation_summary.csv` → currently **111**

Checkpoint resolution (per dataset run directory):

1. `runs/<dataset>/tri_encoder_cfm_v2/checkpoints/epoch_{best_epoch:04d}.pt`
2. Else `checkpoints/best.pt` with a console warning (best-metric epoch may differ from summary row).

Override with `--bijie-checkpoint` / `--l4s-checkpoint` or `--bijie-epoch` / `--l4s-epoch` if your server stores weights elsewhere.

Training hyper-parameters are aligned with `final_metrics.csv` in each run (e.g. `metric_threshold=0.6`, `val_infer_flow_steps=0`, `ctx_ch=4` Bijie / `6` L4S).

---

## 4. How to run (your server)

**Repo layout A** — `Geology-Aware_Hybrid_SAM` (no `SAM/` prefix): `resources/`, `runs/`, `model_architecture_cfm_landseg/` at repo root.

**Repo layout B** — `CSIR_NEIST`: everything under `SAM/`.

```bash
cd ~/Desktop/Deep_learning_projects/CSIR/Geology-Aware_Hybrid_SAM   # your clone

export BIJIE_ROOT=/home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide
export L4S_ROOT=/home/user/Desktop/Deep_learning_projects/4PI/dataset

bash resources/results/run_landslide_presence_eval.sh
```

Requires **`model_architecture_cfm_landseg/eval/eval_landslide_presence.py`** in the repo (not only the shell script). If missing, `git pull` or copy that folder from the branch that added it.

Layout B equivalent: `bash SAM/resources/results/run_landslide_presence_eval.sh` from the CSIR_NEIST root.

Direct Python (layout A):

```bash
python3 model_architecture_cfm_landseg/eval/eval_landslide_presence.py \
  --bijie-root "$BIJIE_ROOT" \
  --l4s-root "$L4S_ROOT" \
  --bijie-summary resources/results/bijie_ablation_report/bijie_best_validation_summary.csv \
  --l4s-summary resources/results/l4s_ablation_report/landslide4sense_best_validation_summary.csv \
  --bijie-run-dir runs/bijie/tri_encoder_cfm_v2 \
  --l4s-run-dir runs/landslide4sense/tri_encoder_cfm_v2 \
  --output-dir resources/results/landslide_presence_report \
  --device cuda \
  --batch-size 16
```

Requirements: PyTorch, same Python path as training (`SAM` on `PYTHONPATH` or run from repo root as above).

---

## 5. Outputs (fill into paper after you run)

| Location | Path |
| --- | --- |
| Working outputs (eval script) | `resources/results/landslide_presence_report/` |
| **Paper bundle (flat, all paper files)** | `model_architecture_cfm_landseg/paper_submission_bundle/` |

After eval: `bash resources/results/populate_paper_submission_bundle.sh`

| File | Purpose |
| --- | --- |
| `tri_encoder_presence_combined_table.csv` | **Single long table** — Bijie + L4S rows (paste into Word / LaTeX) |
| `tri_encoder_presence_run_manifest.csv` | Epoch, checkpoint path, threshold per dataset |
| `tri_encoder_presence_images_bijie.csv` | Per-image diagnostics (Bijie val) |
| `tri_encoder_presence_images_l4s.csv` | Per-image diagnostics (L4S val) |
| `fig_tri_encoder_presence_score_histogram.png` | **Figure**: score distributions for GT landslide vs non-landslide (both datasets) |

### 5.1 Placeholder table (replace with your numbers)

| model_id | dataset | best_epoch | gt_class | n_images | detection_rate | specificity | false_alarm_rate | mean_pred_fg_fraction | mean_pixel_f1 | image_auroc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tri_encoder_cfm_v2 | bijie | 112 | landslide | _run_ | _run_ | — | — | _run_ | _run_ | — |
| tri_encoder_cfm_v2 | bijie | 112 | non_landslide | _run_ | — | _run_ | _run_ | _run_ | _run_ | — |
| tri_encoder_cfm_v2 | bijie | 112 | all_images | _run_ | — | — | — | — | — | _run_ |
| tri_encoder_cfm_v2 | landslide4sense | 111 | landslide | _run_ | _run_ | — | — | _run_ | _run_ | — |
| tri_encoder_cfm_v2 | landslide4sense | 111 | non_landslide | _run_ | — | _run_ | _run_ | _run_ | _run_ | — |
| tri_encoder_cfm_v2 | landslide4sense | 111 | all_images | _run_ | — | — | — | — | — | _run_ |

**Suggested manuscript wording (after numbers exist):**

> We report image-level landslide **presence** on the held-out validation split using the same probability threshold as training (0.6). A tile is predicted positive when at least one connected foreground component (≥ 20 px) appears. On Bijie, non-landslide scenes use an empty reference mask; on Landslide4Sense, negatives are empty-mask competition tiles only. Pixel-level F1/IoU in the main tables quantify mask quality; presence metrics quantify false alarms and missed activations.

### 5.2 Figure caption (histogram)

> **Figure X.** Distribution of image-level detection scores (max instance probability) on the validation split for TriEncoderCFMNet at the best summary epoch, separated by ground-truth landslide presence. Left: Bijie (landslide vs non-landslide folders). Right: Landslide4Sense (mask-positive vs empty-mask tiles). Vertical dashed line: operating threshold 0.6.

---

## 6. Implementation map

| Component | Path |
| --- | --- |
| Evaluation entrypoint | `SAM/model_architecture_cfm_landseg/eval/eval_landslide_presence.py` |
| Shell wrapper | `SAM/resources/results/run_landslide_presence_eval.sh` |
| Model | `TriEncoderCFMNet` |
| Bijie val data | `build_bijie_split` + `BijieTripleStreamDataset` |
| L4S val data | `build_l4s_split` + `L4STripleStreamDataset` |
| Metric parity | `common/metrics.image_level_metrics_from_logits` (same component logic) |

---

## 7. Limitations to disclose

1. **L4S negatives are sparse** if most val tiles contain landslide pixels; report `n_images` for the non-landslide row.
2. **Threshold fixed at 0.6** — same as training logs; optional sweep is future work (`image_best_f1` column hints at optimal score threshold on val).
3. **Validation only** — not test-set generalization unless you extend the script to `test` splits.
4. **Same seed (42) splits** as training — fair comparison but not a fresh external set.

---

## 8. Adding results to the Word manuscript

After you run the script, send back `tri_encoder_presence_combined_table.csv` and the PNG. Then:

1. Insert **Table X** from section 5.1 (filled numbers).
2. Insert **Figure X** from `fig_tri_encoder_presence_score_histogram.png`.
3. Add one paragraph in Results referencing false-alarm rate on Bijie non-landslide and detection rate on landslide images.

`build_conference_docx.py` is **not** auto-updated; paste manually or ask to wire a table builder once numbers exist.

---

_Last updated when the evaluation script was added; numeric cells are placeholders until you run on the server._
