#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export GEPA_MAX_FULL_EVALS="${GEPA_MAX_FULL_EVALS:-16}"
export GEPA_REFLECTION_MINIBATCH_SIZE="${GEPA_REFLECTION_MINIBATCH_SIZE:-16}"
GEPA_CAPTION_SUPERVISION="${GEPA_CAPTION_SUPERVISION:-auto}"

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh" \
  --gepa-caption-supervision "${GEPA_CAPTION_SUPERVISION}" \
  "$@"
