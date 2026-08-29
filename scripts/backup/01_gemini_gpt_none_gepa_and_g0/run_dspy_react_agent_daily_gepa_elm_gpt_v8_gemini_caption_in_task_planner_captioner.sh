#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEPA_EXPERIMENT_NAME="planner_workflow_prompt_and_captioner_default_caption_instruction"
export OPTIMIZE_TARGETS_CSV="planner.workflow_prompt,captioner.default_caption_instruction"
exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh" "$@"

