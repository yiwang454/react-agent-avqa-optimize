#!/usr/bin/env bash
set -euo pipefail

# Original GEPA experiment script:
# scripts/run_dspy_react_agent_daily_gepa_g0_elm_gpt4.1_qwen_gemini_v8_caption_in_task_planner_gpt5_4_reflection.sh
#
# This wrapper retains the original GEPA runner's runtime configuration and
# enables its inference-only branch. Results resume in the original GEPA run
# directory; a completed per-question cache entry is never rerun.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_SCRIPT="${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_g0_elm_gpt4.1_qwen_gemini_v8_caption_in_task_planner_gpt5_4_reflection.sh"

# REQUIRED: paste the Qwen OpenAI-compatible /v1 endpoint between the quotes.
# This value overrides the URL configured by the original script and its YAML.
QWEN_BASE_URL_OVERRIDE="http://10.62.186.37:8000/v1"
if [ -z "${QWEN_BASE_URL_OVERRIDE}" ]; then
  echo "Set QWEN_BASE_URL_OVERRIDE in this script before running." >&2
  exit 2
fi

export QWEN_BASE_URL_OVERRIDE
export INFERENCE_ONLY=true
export FINAL_EVAL_NUM_THREADS=4
export QWEN_RELIABLE_MAX_BATCH_RETRIES=3
export QWEN_RELIABLE_FALLBACK_NUM_THREADS=2
exec "${ORIGINAL_SCRIPT}" "$@"
