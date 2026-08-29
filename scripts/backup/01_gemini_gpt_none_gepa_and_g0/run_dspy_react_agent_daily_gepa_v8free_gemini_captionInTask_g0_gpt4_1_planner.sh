#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEPA_EXPERIMENT_NAME="v8free_g0_planner_gpt4_1"
export GEPA_SUPERVISION_TIER="G0"
export PLANNER_MODEL_OVERRIDE="gpt-4.1"

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_common.sh" "$@"
