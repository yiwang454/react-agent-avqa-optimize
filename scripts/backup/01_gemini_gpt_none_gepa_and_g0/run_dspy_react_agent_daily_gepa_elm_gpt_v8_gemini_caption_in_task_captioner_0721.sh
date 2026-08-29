#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEPA_EXPERIMENT_NAME="${GEPA_EXPERIMENT_NAME:-captioner_default_caption_instruction_0721}"
export OPTIMIZE_TARGETS_CSV="${OPTIMIZE_TARGETS_CSV:-captioner.default_caption_instruction}"
export GEPA_CAPTION_SUPERVISION="${GEPA_CAPTION_SUPERVISION:-none}"

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common_0721.sh" "$@"
