#!/usr/bin/env bash
set -euo pipefail

# G1: optimize only the GPT-4.1 captioner instruction with the concatenated
# AV-alignment, video-consistent, and audio-revised caption evidence.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export GEPA_EXPERIMENT_NAME="captioner_default_caption_instruction_gpt4_1_g1_privileged"
export OPTIMIZE_TARGETS_CSV="captioner.default_caption_instruction"
export GEPA_CAPTION_SUPERVISION="privileged"

export PLANNER_MODEL_OVERRIDE="${PLANNER_MODEL_OVERRIDE:-gpt-4.1}"
export GEPA_REFLECTION_MODEL="${GEPA_REFLECTION_MODEL_OVERRIDE:-gpt-4.1}"
export GEPA_REFLECTION_TEMPERATURE="${GEPA_REFLECTION_TEMPERATURE_OVERRIDE:-0.0}"
export GEPA_REFLECTION_REASONING_EFFORT="${GEPA_REFLECTION_REASONING_EFFORT_OVERRIDE:-none}"

# Select the captioner-specific template from the current non-densified G1
# template set; PromptTargetGEPAAdapter selects its captioner entry by target.
export GEPA_REFLECTION_TEMPLATE_YAML="${GEPA_REFLECTION_TEMPLATE_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/DSPy/reflection_template.yaml}"
export GEPA_REFLECTION_TEMPLATE_VERSION="${GEPA_REFLECTION_TEMPLATE_VERSION_OVERRIDE:-original}"

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common_0721.sh" "$@"
