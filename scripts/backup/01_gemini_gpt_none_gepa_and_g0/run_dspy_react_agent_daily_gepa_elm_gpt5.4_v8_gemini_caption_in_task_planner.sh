#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEPA_EXPERIMENT_NAME="planner_workflow_prompt_gpt5_4_medium"
export OPTIMIZE_TARGETS_CSV="planner.workflow_prompt"
# Keep the existing full-evaluation budget (8), while making each reflective
# comparison more representative than the former 8-example minibatch.
export GEPA_REFLECTION_MINIBATCH_SIZE="${GEPA_REFLECTION_MINIBATCH_SIZE:-16}"

export PLANNER_MODEL_OVERRIDE="${PLANNER_MODEL_OVERRIDE:-gpt-5.4}"
export PLANNER_REASONING_EFFORT="${PLANNER_REASONING_EFFORT_OVERRIDE:-medium}"
export GEPA_REFLECTION_MODEL="${GEPA_REFLECTION_MODEL_OVERRIDE:-gpt-5.4}"
export GEPA_REFLECTION_TEMPERATURE="${GEPA_REFLECTION_TEMPERATURE_OVERRIDE:-0.0}"
export GEPA_REFLECTION_REASONING_EFFORT="${GEPA_REFLECTION_REASONING_EFFORT_OVERRIDE:-medium}"
export GEPA_MAX_FULL_EVALS="${GEPA_MAX_FULL_EVALS:-16}"
export GEPA_REFLECTION_MINIBATCH_SIZE="${GEPA_REFLECTION_MINIBATCH_SIZE:-16}"

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh" "$@"
