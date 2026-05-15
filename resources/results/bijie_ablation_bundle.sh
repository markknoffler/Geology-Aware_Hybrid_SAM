#!/usr/bin/env bash
# Bijie ablation: build the same *_best_validation_summary.csv style as Landslide4Sense,
# then generate the same paper-style PNG bundles (fig01–fig09 + conference Fig02–Fig13).
#
# Prerequisites: epoch_metrics*.csv under
#   SAM/ablation_study/baseline_models/*/bijie/*/results/
#   SAM/ablation_study/dual_stream_gated/outputs_bijie/**/
#   SAM/runs/bijie/<experiment>/results/
#
# This script is not executed automatically. From the repository root (CSIR_NEIST):
#
#   bash SAM/resources/results/bijie_ablation_bundle.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-python3}"
REPORT="${ROOT}/SAM/resources/results/build_l4s_ablation_report.py"
GENFIG="${ROOT}/SAM/resources/results/generate_paper_comparison_figures.py"
OUT_CSV_DIR="${ROOT}/SAM/resources/results/bijie_ablation_report"
SUM_CSV="${OUT_CSV_DIR}/bijie_best_validation_summary.csv"
FIG_OUT="${OUT_CSV_DIR}/paper_comparison_figures"

echo "== Bijie summary CSV (best epoch per model, dual_stream_gated above tri_encoder) =="
"${PY}" "${REPORT}" --dataset bijie --output-dir "${OUT_CSV_DIR}"

echo "== Bijie paper figures (generic fig01–fig09 + conference_bijie/Fig02–Fig13) =="
"${PY}" "${GENFIG}" \
  --summary-csv "${SUM_CSV}" \
  --output-dir "${FIG_OUT}" \
  --focus-model-id tri_encoder_cfm_v2

echo "Done. Summary: ${SUM_CSV}"
echo "Figures: ${FIG_OUT}/ and ${FIG_OUT}/conference_bijie/"
