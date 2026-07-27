#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DENSIFIED_LABEL_DIR="${GEPA_DENSIFIED_LABEL_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_densified_labels_selectedTrain125_gpt-5.4_high}"

# These experiments use only the compact per-question labels, not the older
# full three-caption privileged supervision path.
export GEPA_CAPTION_SUPERVISION="none"

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common_0721.sh" \
  --gepa-densified-label-dir "${DENSIFIED_LABEL_DIR}" \
  "$@"
