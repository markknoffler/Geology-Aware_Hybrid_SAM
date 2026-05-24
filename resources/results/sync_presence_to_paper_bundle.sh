#!/usr/bin/env bash
# Deprecated: use populate_paper_submission_bundle.sh (flat bundle, no subfolders).
exec "$(dirname "$0")/populate_paper_submission_bundle.sh" "$@"
