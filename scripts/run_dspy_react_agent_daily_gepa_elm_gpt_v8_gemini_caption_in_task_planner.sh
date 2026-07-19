#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEPA_EXPERIMENT_NAME="planner_workflow_prompt"
export OPTIMIZE_TARGETS_CSV="planner.workflow_prompt"
# Keep the existing full-evaluation budget (8), while making each reflective
# comparison more representative than the former 8-example minibatch.
export GEPA_REFLECTION_MINIBATCH_SIZE="${GEPA_REFLECTION_MINIBATCH_SIZE:-16}"
exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh" "$@"

