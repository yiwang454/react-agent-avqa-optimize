#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GEPA_EXPERIMENT_NAME="${GEPA_EXPERIMENT_NAME:-densified_planner_workflow_prompt_gpt4_1}"
export OPTIMIZE_TARGETS_CSV="planner.workflow_prompt"
export PLANNER_MODEL_OVERRIDE="gpt-4.1"
export PLANNER_REASONING_EFFORT_OVERRIDE="none"
export GEPA_REFLECTION_MODEL="gpt-4.1"
export GEPA_REFLECTION_REASONING_EFFORT="none"
export GEPA_REFLECTION_TEMPERATURE="0.0"

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_densified_common.sh" "$@" >> "${SCRIPT_DIR}/logs/${GEPA_EXPERIMENT_NAME}.log" 2>&1