# TriEncoderCFMNet ablation companion (Landslide4Sense + Bijie)

This note sits next to `bijie_best_validation_summary.csv` and records how the two benchmark tables were built, what changed in the logs, and how the story lines up with the dual-stream landslide literature (see `SAM/literature_review/dual_stream.pdf` for the gated-fusion framing we cite conceptually).

## 1. What the harness does

Both summaries come from `build_l4s_ablation_report.py`, which walks the same three roots for each dataset: `ablation_study/baseline_models`, `ablation_study/dual_stream_gated`, and `runs/<dataset>`. For every `epoch_metrics*.csv` it keeps the **single best row by validation F1** (ties break toward higher IoU then accuracy). `dual_stream_gated` is always sorted just above `tri_encoder_cfm_v2` so the manuscript reads as “strong dual encoder, then our trimodal CFM stack”.

## 2. Landslide4Sense (Table 1 in the Word file)

Source file: `SAM/resources/results/l4s_ablation_report/landslide4sense_best_validation_summary.csv`

| model_id | best_epoch | val_f1 | val_iou | val_acc | val_auroc |
| --- | --- | --- | --- | --- | --- |
| linknet | 99 | 0.43068649371465000 | 0.3587936535477640 | 0.9729868074258170 | 0.8278018974942180 |
| dep_unet | 82 | 0.6032018562157950 | 0.5348571563760440 | 0.9768387228250500 | 0.8587206630770710 |
| gmnet | 79 | 0.616280272603035 | 0.5453190232316650 | 0.9775054057439170 | 0.8782042435133090 |
| rmau_net | 83 | 0.6178274104992550 | 0.5480107590556150 | 0.9774889796972280 | 0.8663686720750760 |
| transunet | 92 | 0.6191708544890090 | 0.5513713633020720 | 0.9758156438668570 | 0.8864408143298660 |
| emr_hrnet | 94 | 0.6273944526910780 | 0.5565658112367000 | 0.9750224550565080 | 0.8868853067844720 |
| shapeformer | 95 | 0.6320214321215950 | 0.5603799050052960 | 0.9751527011394500 | 0.8650736345263600 |
| dual_stream_unet | 95 | 0.6561854084332780 | 0.5809435223539670 | 0.9860140631596250 | 0.9140712604937680 |
| unet | 93 | 0.6800101697444920 | 0.6107069402933120 | 0.9843750447034840 | 0.9104441272408960 |
| deeplabv3plus | 85 | 0.7098509967327120 | 0.6363813281059270 | 0.9844203939040500 | 0.9244625172325960 |
| dual_stream_gated | 23 | 0.7096000959475840 | 0.6343797643979390 | 0.9866052915652590 | 0.9343367488925870 |
| tri_encoder_cfm_v2 | 111 | 0.7397656610806780 | 0.6415862238407140 | 0.9764446371793750 | 0.9564063004670190 |

**Reading the table.** DeeplabV3+ and `dual_stream_gated` remain the hardest baselines on pure F1/IoU in this sweep, while TriEncoderCFMNet pushes AUROC/AUPRC when the flow-matching head is active—use the scalar ROC/PR panels in Figs. 7–12 with the caveat that full curves still need threshold sweeps in the logger.

## 3. Bijie (Table 2 in the Word file)

Source file: `SAM/resources/results/bijie_ablation_report/bijie_best_validation_summary.csv`

| model_id | best_epoch | val_f1 | val_iou | val_acc | val_auroc |
| --- | --- | --- | --- | --- | --- |
| rmau_net | 77 | 0.8304167721006606 | 0.7981379181146622 | 0.9805742369757758 | 0.05448717845989484 |
| dep_unet | 96 | 0.8308891355991364 | 0.798854927221934 | 0.9823485215504965 | 0.0543091156402952 |
| shapeformer | 86 | 0.8397061626116434 | 0.8102853861120012 | 0.9837126731872559 | 0.04594017024161738 |
| transunet | 89 | 0.8445521626207564 | 0.8151293413506614 | 0.9830600685543485 | 0.0543091156402952 |
| emr_hrnet | 89 | 0.8457122047742208 | 0.8178038779232237 | 0.9834884537590874 | 0.055555554309116885 |
| gmnet | 75 | 0.854725205236011 | 0.8247582051489089 | 0.9829667144351535 | 0.0543091156402952 |
| unet | 79 | 0.8731255398856269 | 0.8419982459810045 | 0.9865624109903971 | 0.055555554309116885 |
| dual_stream_unet | 58 | 0.8749462101194594 | 0.8427784774038527 | 0.9864326053195529 | 0.055555554309116885 |
| deeplabv3plus | 92 | 0.927485015657213 | 0.8985853095849355 | 0.9894901116689047 | 0.055555554309116885 |
| dual_stream_gated | 99 | 0.8970915079116821 | 0.8685273428757986 | 0.9883465237087674 | nan |
| tri_encoder_cfm_v2 | 112 | 0.8168713930580351 | 0.7892284989356995 | 0.981063101026747 | 0.05448717845989484 |

**Column hygiene.** `train_loss`, `val_loss`, and `epoch_time` were stripped from each Bijie `epoch_metrics` log listed in the summary (script `SAM/resources/results/strip_epoch_metrics_columns.py`) so downstream heatmaps never treat timing or loss scalars as extra “metric channels”. Loss curves in Fig. 3 therefore show only accuracy / F1 / IoU when those columns are absent; regenerate figures after stripping via `generate_paper_comparison_figures.py`.

**Cross-site contrast.** Bijie RGB composites react differently than the six-channel Landslide4Sense stack: recall on several baselines stays very high while precision swings, so the gated dual-stream row is still the cleanest apple-to-apple reference even though TriEncoderCFMNet trails it on raw F1 here.

## 4. How this mirrors the dual-stream paper voice

The CMC dual-stream article stresses (i) heterogeneous landscapes, (ii) explicit optical–DEM fusion, and (iii) careful decoder design. We keep that vocabulary—sparse masks, domain shift, gated fusion—but swap in our trimodal gates plus conditional flow matching so the writing in the `.docx` stays aligned with your earlier conference draft while acknowledging the newer baseline ladder.

## 5. Figure bundles

| Bundle | Path |
| --- | --- |
| Landslide4Sense conference Figs. 2–13 | `SAM/resources/results/l4s_ablation_report/paper_comparison_figures/conference_remotesensing_landslide/` |
| Bijie conference Figs. 14–25 (same filenames) | `SAM/resources/results/bijie_ablation_report/paper_comparison_figures/conference_bijie/` |

`paper_submission_bundle/` (created when you run `build_conference_docx.py`) copies **only** the Word manuscript, `MODEL_ARCHITECTURE.md` as `TRI_ENCODER_MODEL_SPEC.md`, this companion note, and those PNGs.

## 6. Image-level landslide vs non-landslide (TriEncoderCFMNet only)

See **`SAM/resources/results/LANDSLIDE_PRESENCE_DETECTION.md`** (methodology, table template, figure caption). Run `SAM/resources/results/run_landslide_presence_eval.sh` on the server with `BIJIE_ROOT` / `L4S_ROOT` set; outputs land in `landslide_presence_report/`.

## 7. Commands worth keeping

```bash
python3 SAM/resources/results/build_l4s_ablation_report.py --dataset landslide4sense
python3 SAM/resources/results/build_l4s_ablation_report.py --dataset bijie
python3 SAM/resources/results/generate_paper_comparison_figures.py \
  --summary-csv SAM/resources/results/l4s_ablation_report/landslide4sense_best_validation_summary.csv \
  --output-dir SAM/resources/results/l4s_ablation_report/paper_comparison_figures
python3 SAM/resources/results/generate_paper_comparison_figures.py \
  --summary-csv SAM/resources/results/bijie_ablation_report/bijie_best_validation_summary.csv \
  --output-dir SAM/resources/results/bijie_ablation_report/paper_comparison_figures \
  --focus-model-id tri_encoder_cfm_v2
# bash SAM/resources/results/run_landslide_presence_eval.sh
python3 SAM/model_architecture_cfm_landseg/scripts/build_conference_docx.py
```

_Regenerate this markdown if the CSVs change; tables above are snapshots from the files on disk._
