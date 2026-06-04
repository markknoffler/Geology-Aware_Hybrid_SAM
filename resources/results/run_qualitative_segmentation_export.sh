#!/usr/bin/env bash
# Export 11 qualitative samples per folder (image + GT mask + predicted mask).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root: .../SAM/resources/results -> .../SAM -> parent may be Geology-Aware_Hybrid_SAM
SAM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
if [[ -d "${SAM_ROOT}/model_architecture_cfm_landseg" ]]; then
  REPO_ROOT="${SAM_ROOT}"
else
  REPO_ROOT="$(cd "${SAM_ROOT}/.." && pwd)"
fi

BIJIE_ROOT="${BIJIE_ROOT:-/home/user/Desktop/Deep_learning_projects/4PI/dataset_bijie_landslide}"
L4S_ROOT="${L4S_ROOT:-/home/user/Desktop/Deep_learning_projects/4PI/dataset}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/model_architecture_cfm_landseg/paper_submission_bundle/qualitative_segmentation_11}"
BIJIE_CKPT="${BIJIE_CKPT:-${REPO_ROOT}/runs/bijie/tri_encoder_cfm_v2/checkpoint/best.pt}"
L4S_CKPT="${L4S_CKPT:-${REPO_ROOT}/runs/landslide4sense/tri_encoder_cfm_v2/checkpoint/best.pt}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/ablation_study/baseline_models:${PYTHONPATH:-}"

python3 "${REPO_ROOT}/model_architecture_cfm_landseg/eval/export_qualitative_segmentation_samples.py" \
  --bijie-root "${BIJIE_ROOT}" \
  --l4s-root "${L4S_ROOT}" \
  --bijie-checkpoint "${BIJIE_CKPT}" \
  --l4s-checkpoint "${L4S_CKPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --num-samples "${NUM_SAMPLES:-11}" \
  --seed "${SEED:-42}" \
  --device "${DEVICE:-cuda}"

echo "Wrote qualitative export to: ${OUTPUT_DIR}"
