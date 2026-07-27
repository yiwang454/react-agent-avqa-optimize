#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEPA_EXPERIMENT_NAME="v8free_g3_key_evidence_planner_gpt4_1"
export GEPA_SUPERVISION_TIER="G3"
export PLANNER_MODEL_OVERRIDE="gpt-4.1"

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_common.sh" "$@" >> "${SCRIPT_DIR}/logs/${GEPA_EXPERIMENT_NAME}.log" 2>&1
