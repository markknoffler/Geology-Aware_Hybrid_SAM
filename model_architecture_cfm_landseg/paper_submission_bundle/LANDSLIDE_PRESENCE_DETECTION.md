# Image-level landslide vs non-landslide evaluation (TriEncoderCFMNet)

**Model:** `tri_encoder_cfm_v2`  
**All files live in this same folder:** `paper_submission_bundle/` (flat — no subdirectories).  
**Checkpoints:** `runs/*/tri_encoder_cfm_v2/checkpoint/best.pt` (summary CSV cites epochs 112 Bijie / 111 L4S).

---

## 1. Why we did this

Pixel summary CSVs (`landslide4sense_best_validation_summary.csv`, `bijie_best_validation_summary.csv`) measure **mask quality**. This pass measures **image-level presence**: does the model activate on landslide scenes and stay silent on non-landslide scenes (empty GT mask)?

---

## 2. What we did

| Item | Detail |
| --- | --- |
| Bijie val | 554 images: 154 landslide + 400 non-landslide (`non-landslide/image`, zero mask) |
| L4S val | 380 images: 220 mask-positive + 160 empty-mask tiles |
| Rule | Prob ≥ 0.6, ≥1 connected component ≥ 20 px → predicted landslide |
| Script | `model_architecture_cfm_landseg/eval/eval_landslide_presence.py` |

---

## 3. Results

From `tri_encoder_presence_combined_table.csv` in this folder.

### Bijie

| GT class | *n* | Key metric | Value |
| --- | ---: | --- | --- |
| Landslide | 154 | Detection rate | **95.5%** (7 missed) |
| Non-landslide | 400 | Specificity | **87.8%** |
| Non-landslide | 400 | False alarm rate | **12.3%** |
| All images | 554 | Image AUROC | **0.956** |

### Landslide4Sense

| GT class | *n* | Key metric | Value |
| --- | ---: | --- | --- |
| Landslide | 220 | Detection rate | **90.5%** (21 missed) |
| Non-landslide | 160 | Specificity | **87.5%** |
| Non-landslide | 160 | False alarm rate | **12.5%** |
| All images | 380 | Image AUROC | **0.909** |

**Figure:** `fig_tri_encoder_presence_score_histogram.png`

---

## 4. Files in `paper_submission_bundle/` (this directory)

| File | Role |
| --- | --- |
| `LANDSLIDE_PRESENCE_DETECTION.md` | This document |
| `tri_encoder_presence_combined_table.csv` | Paper table (both datasets) |
| `tri_encoder_presence_run_manifest.csv` | Checkpoint paths used |
| `tri_encoder_presence_images_bijie.csv` | Per-image Bijie |
| `tri_encoder_presence_images_l4s.csv` | Per-image L4S |
| `fig_tri_encoder_presence_score_histogram.png` | Presence figure |
| `landslide4sense_best_validation_summary.csv` | Pixel ablation L4S |
| `bijie_best_validation_summary.csv` | Pixel ablation Bijie |
| `BIJIE_AND_L4S_ABLATION_COMPANION.md` | Ablation narrative |
| `TRI_ENCODER_MODEL_SPEC.md` | Architecture spec |
| `l4s_Fig02_…` … `l4s_Fig13_…` | L4S conference figures |
| `bijie_Fig02_…` … `bijie_Fig13_…` | Bijie conference figures |
| `overlay_fig01_…` … `overlay_fig09_…` | Supplementary overlays (no fig05) |
| `conference_manuscript_tri_encoder_cfm.docx` | Manuscript |

Refresh everything flat: `bash resources/results/populate_paper_submission_bundle.sh`
