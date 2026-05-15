#!/usr/bin/env bash
# Run TriEncoderCFMNet image-level landslide vs non-landslide evaluation.
#
# Supports both repo layouts:
#   A) .../SAM/resources/results/run_landslide_presence_eval.sh  (CSIR_NEIST)
#   B) .../resources/results/run_landslide_presence_eval.sh      (Geology-Aware_Hybrid_SAM)
#
# From repo root:
#   export BIJIE_ROOT=/home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide
#   export L4S_ROOT=/home/user/Desktop/Deep_learning_projects/4PI/dataset
#   bash resources/results/run_landslide_presence_eval.sh
#   # or: bash SAM/resources/results/run_landslide_presence_eval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Repo root: two levels up from resources/results
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Code root: directory that contains model_architecture_cfm_landseg + ablation_study + runs
if [[ -d "${REPO_ROOT}/SAM/model_architecture_cfm_landseg" ]]; then
  CODE_ROOT="${REPO_ROOT}/SAM"
elif [[ -d "${REPO_ROOT}/model_architecture_cfm_landseg" ]]; then
  CODE_ROOT="${REPO_ROOT}"
else
  echo "ERROR: Cannot find model_architecture_cfm_landseg under ${REPO_ROOT} or ${REPO_ROOT}/SAM" >&2
  exit 1
fi

EVAL_PY="${CODE_ROOT}/model_architecture_cfm_landseg/eval/eval_landslide_presence.py"
if [[ ! -f "${EVAL_PY}" ]]; then
  echo "ERROR: Missing ${EVAL_PY}" >&2
  echo "Pull/sync model_architecture_cfm_landseg/eval/ from the repo (not only this shell script)." >&2
  exit 1
fi

BIJIE_ROOT="${BIJIE_ROOT:-/home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide}"
L4S_ROOT="${L4S_ROOT:-/home/user/Desktop/Deep_learning_projects/4PI/dataset}"

RESULTS_DIR="${SCRIPT_DIR}"
BIJIE_SUMMARY="${RESULTS_DIR}/bijie_ablation_report/bijie_best_validation_summary.csv"
L4S_SUMMARY="${RESULTS_DIR}/l4s_ablation_report/landslide4sense_best_validation_summary.csv"
OUT_DIR="${RESULTS_DIR}/landslide_presence_report"

python3 "${EVAL_PY}" \
  --bijie-root "${BIJIE_ROOT}" \
  --l4s-root "${L4S_ROOT}" \
  --bijie-summary "${BIJIE_SUMMARY}" \
  --l4s-summary "${L4S_SUMMARY}" \
  --bijie-run-dir "${CODE_ROOT}/runs/bijie/tri_encoder_cfm_v2" \
  --l4s-run-dir "${CODE_ROOT}/runs/landslide4sense/tri_encoder_cfm_v2" \
  --output-dir "${OUT_DIR}" \
  "$@"
