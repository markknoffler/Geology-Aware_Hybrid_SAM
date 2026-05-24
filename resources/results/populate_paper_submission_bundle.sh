#!/usr/bin/env bash
# Copy all paper-related CSVs, PNGs, and MD files into ONE flat folder (no subdirectories).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -d "${SCRIPT_DIR}/../../model_architecture_cfm_landseg" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  BUNDLE="${REPO_ROOT}/model_architecture_cfm_landseg/paper_submission_bundle"
  R="${REPO_ROOT}/resources/results"
elif [[ -d "${SCRIPT_DIR}/../SAM/model_architecture_cfm_landseg" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  BUNDLE="${REPO_ROOT}/SAM/model_architecture_cfm_landseg/paper_submission_bundle"
  R="${REPO_ROOT}/SAM/resources/results"
else
  echo "ERROR: cannot locate model_architecture_cfm_landseg" >&2
  exit 1
fi

mkdir -p "${BUNDLE}"

# --- Markdown (paper docs only) ---
cp -f "${R}/bijie_ablation_report/BIJIE_AND_L4S_ABLATION_COMPANION.md" "${BUNDLE}/" 2>/dev/null || true
if [[ -f "${BUNDLE}/../MODEL_ARCHITECTURE.md" ]]; then
  cp -f "${BUNDLE}/../MODEL_ARCHITECTURE.md" "${BUNDLE}/TRI_ENCODER_MODEL_SPEC.md"
fi
if [[ -f "${R}/LANDSLIDE_PRESENCE_DETECTION.md" ]]; then
  cp -f "${R}/LANDSLIDE_PRESENCE_DETECTION.md" "${BUNDLE}/"
elif [[ -f "${BUNDLE}/landslide_presence/LANDSLIDE_PRESENCE_DETECTION.md" ]]; then
  cp -f "${BUNDLE}/landslide_presence/LANDSLIDE_PRESENCE_DETECTION.md" "${BUNDLE}/"
fi

# --- Summary + presence CSVs ---
for f in \
  "${R}/l4s_ablation_report/landslide4sense_best_validation_summary.csv" \
  "${R}/bijie_ablation_report/bijie_best_validation_summary.csv" \
  "${R}/landslide_presence_report/tri_encoder_presence_combined_table.csv" \
  "${R}/landslide_presence_report/tri_encoder_presence_run_manifest.csv" \
  "${R}/landslide_presence_report/tri_encoder_presence_images_bijie.csv" \
  "${R}/landslide_presence_report/tri_encoder_presence_images_l4s.csv"
do
  [[ -f "$f" ]] && cp -f "$f" "${BUNDLE}/"
done

# Legacy nested presence csv (if user has not re-run populate before)
for f in "${BUNDLE}"/landslide_presence/csv/*.csv; do
  [[ -f "$f" ]] && cp -f "$f" "${BUNDLE}/"
done

# --- L4S conference figures (prefix l4s_) ---
L4S_CONF="${R}/l4s_ablation_report/paper_comparison_figures/conference_remotesensing_landslide"
if [[ -d "${L4S_CONF}" ]]; then
  for p in "${L4S_CONF}"/Fig*.png "${L4S_CONF}"/Fig*.txt; do
    [[ -f "$p" ]] && cp -f "$p" "${BUNDLE}/l4s_${p##*/}"
  done
fi

# --- Bijie conference figures (prefix bijie_) ---
BIJIE_CONF="${R}/bijie_ablation_report/paper_comparison_figures/conference_bijie"
if [[ -d "${BIJIE_CONF}" ]]; then
  for p in "${BIJIE_CONF}"/Fig*.png "${BIJIE_CONF}"/Fig*.txt; do
    [[ -f "$p" ]] && cp -f "$p" "${BUNDLE}/bijie_${p##*/}"
  done
fi

# --- L4S overlay figures (prefix overlay_) ---
OVERLAY="${R}/l4s_ablation_report/paper_comparison_figures"
if [[ -d "${OVERLAY}" ]]; then
  for p in "${OVERLAY}"/fig*.png; do
    [[ -f "$p" ]] || continue
    base="${p##*/}"
    [[ "$base" == "fig05_"* ]] && continue
    cp -f "$p" "${BUNDLE}/overlay_${base}"
  done
fi

# --- Presence histogram ---
if [[ -f "${R}/landslide_presence_report/fig_tri_encoder_presence_score_histogram.png" ]]; then
  cp -f "${R}/landslide_presence_report/fig_tri_encoder_presence_score_histogram.png" "${BUNDLE}/"
fi

# Remove old nested layout (flat bundle only)
rm -rf "${BUNDLE}/landslide_presence" "${BUNDLE}/figures_l4s" "${BUNDLE}/figures_bijie" "${BUNDLE}/figures_overlay_l4s"

echo "Flat paper bundle: ${BUNDLE}"
ls -la "${BUNDLE}"
