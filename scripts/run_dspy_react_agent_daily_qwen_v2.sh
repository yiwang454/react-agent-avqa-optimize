#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_qwen_v2}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

# conda activate "${ENV_PREFIX}"
cd "${PROJECT_DIR}"

# Print related settings to verify env loading without leaking secrets.
echo "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:+***set***}"
echo "DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL}"
echo "DEEPSEEK_MODEL=${DEEPSEEK_MODEL}"
echo "PLANNER_THINKING_MODE=${PLANNER_THINKING_MODE:-}"
echo "PLANNER_TEMPERATURE=${PLANNER_TEMPERATURE:-}"
echo "PLANNER_TOP_P=${PLANNER_TOP_P:-}"
echo "QWEN_API_KEY=${QWEN_API_KEY:+***set***}"
echo "QWEN_BASE_URL=${QWEN_BASE_URL:-}"
echo "QWEN_MODEL=${QWEN_MODEL:-}"
echo "QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-}"
echo "PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML:-}"
echo "PROMPT_YAML=${PROMPT_YAML:-}"
echo "DSPY_AVQA_ALLOWED_TOOLS=${DSPY_AVQA_ALLOWED_TOOLS:-}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "AUDIO_CAPTION_DIR=${AUDIO_CAPTION_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "CONCURRENCY=${CONCURRENCY}"
echo "DEBUG=${DEBUG}"
echo "DEBUG_LIMIT=${DEBUG_LIMIT}"

RUN_REPEATS=2
# RUN_SET_TAG="${RUN_SET_TAG:-repeat}"
BASE_OUTPUT_DIR="${OUTPUT_DIR%/}"

echo "RUN_REPEATS=${RUN_REPEATS}"
echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"

for run_idx in $(seq 1 "${RUN_REPEATS}"); do
  run_output_dir="${BASE_OUTPUT_DIR}/repeat_${run_idx}"
  run_output_jsonl="${run_output_dir}/output_test.jsonl"

  echo "Running DSPy batch runner: run ${run_idx}/${RUN_REPEATS}"
  echo "RUN_OUTPUT_DIR=${run_output_dir}"
  echo "RUN_OUTPUT_JSONL=${run_output_jsonl}"

  cmd=(
    python DSPy/avqa_dspy_impl.py
    --input-jsonl "${INPUT_JSONL}"
    --audio-caption-dir "${AUDIO_CAPTION_DIR}"
    --output-jsonl "${run_output_jsonl}"
    --output-dir "${run_output_dir}"
    --max-turns "${MAX_TURNS}"
    --concurrency "${CONCURRENCY}"
    --perception-model "${PERCEPTION_MODEL}"
  )

  if [ -n "${PERCEPTION_CONFIG_YAML:-}" ]; then
    cmd+=(--perception-config-yaml "${PERCEPTION_CONFIG_YAML}")
  fi

  if [ -n "${PROMPT_YAML:-}" ]; then
    cmd+=(--prompt-yaml "${PROMPT_YAML}")
  fi

  if [ -n "${DSPY_AVQA_ALLOWED_TOOLS:-}" ]; then
    cmd+=(--allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}")
  fi

  if [ "${DEBUG}" = "true" ]; then
    cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
  fi

  cmd+=("$@")
  "${cmd[@]}"
done
