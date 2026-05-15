#!/usr/bin/env bash
# Run TriEncoderCFMNet image-level landslide vs non-landslide evaluation on your server.
# Edit DATASET paths below, then:
#   bash SAM/resources/results/run_landslide_presence_eval.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

BIJIE_ROOT="${BIJIE_ROOT:-/path/to/Bijie-landslide-dataset}"
L4S_ROOT="${L4S_ROOT:-/path/to/Landslide4Sense}"

python3 "${ROOT}/SAM/model_architecture_cfm_landseg/eval/eval_landslide_presence.py" \
  --bijie-root "${BIJIE_ROOT}" \
  --l4s-root "${L4S_ROOT}" \
  --bijie-summary "${ROOT}/SAM/resources/results/bijie_ablation_report/bijie_best_validation_summary.csv" \
  --l4s-summary "${ROOT}/SAM/resources/results/l4s_ablation_report/landslide4sense_best_validation_summary.csv" \
  --bijie-run-dir "${ROOT}/SAM/runs/bijie/tri_encoder_cfm_v2" \
  --l4s-run-dir "${ROOT}/SAM/runs/landslide4sense/tri_encoder_cfm_v2" \
  --output-dir "${ROOT}/SAM/resources/results/landslide_presence_report" \
  "$@"
