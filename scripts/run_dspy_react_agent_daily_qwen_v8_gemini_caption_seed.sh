#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_qwen_v8}"

set -a
source "${ENV_FILE}"
set +a

# conda activate "${ENV_PREFIX}"
export PROJECT_DIR="${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"
cd "${PROJECT_DIR}"

# This experiment fixes the Qwen perception seed and DeepSeek planner seed from bash/env.
# Captioning is moved to Gemini to avoid the current Qwen caption response issue.
export QWEN_SEED="${QWEN_SEED:-1234}"
export DEEPSEEK_SEED="${DEEPSEEK_SEED:-7}"
export PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8.yaml}"
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/config_localqwen_api_instruct.yaml}"
export CAPTIONER_MODEL="${CAPTIONER_MODEL_OVERRIDE:-gemini}"
export CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml}"
export DSPY_AVQA_ALLOWED_TOOLS="${DSPY_AVQA_ALLOWED_TOOLS_OVERRIDE:-ask_caption,ask_perception}"


BASE_ENV_OUTPUT_DIR=/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_qwen_v8GeminiCaption
export OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${BASE_ENV_OUTPUT_DIR}_qwen_seed_${QWEN_SEED}_deepseek_seed_${DEEPSEEK_SEED}}"
export OUTPUT_JSONL="${OUTPUT_DIR}/output_test.jsonl"
export QWEN_BASE_URL="http://10.62.187.79:8000/v1"

# Print related settings to verify env loading without leaking secrets.
echo "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:+***set***}"
echo "DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL}"
echo "DEEPSEEK_MODEL=${DEEPSEEK_MODEL}"
echo "DEEPSEEK_SEED=${DEEPSEEK_SEED}"
echo "PLANNER_THINKING_MODE=${PLANNER_THINKING_MODE:-}"
echo "PLANNER_TEMPERATURE=${PLANNER_TEMPERATURE:-}"
echo "PLANNER_TOP_P=${PLANNER_TOP_P:-}"
echo "QWEN_API_KEY=${QWEN_API_KEY:+***set***}"
echo "QWEN_BASE_URL=${QWEN_BASE_URL:-}"
echo "QWEN_MODEL=${QWEN_MODEL:-}"
echo "QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-}"
echo "QWEN_STREAM=${QWEN_STREAM:-}"
echo "QWEN_SEED=${QWEN_SEED}"
echo "CAPTIONER_MODEL=${CAPTIONER_MODEL:-}"
echo "PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML:-}"
echo "CAPTIONER_CONFIG_YAML=${CAPTIONER_CONFIG_YAML:-}"
echo "PROMPT_YAML=${PROMPT_YAML:-}"
echo "DSPY_AVQA_ALLOWED_TOOLS=${DSPY_AVQA_ALLOWED_TOOLS:-}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "AUDIO_CAPTION_DIR=${AUDIO_CAPTION_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "DEBUG=${DEBUG}"
echo "DEBUG_LIMIT=${DEBUG_LIMIT}"

RUN_REPEATS="${RUN_REPEATS:-3}"
BASE_OUTPUT_DIR="${OUTPUT_DIR%/}"
child_pids=()

cleanup_children() {
  local reason="${1:-exit}"
  trap - INT TERM EXIT

  if [ "${#child_pids[@]}" -gt 0 ]; then
    echo "Stopping ${#child_pids[@]} background run(s) after ${reason}..." >&2
    kill -TERM "${child_pids[@]}" 2>/dev/null || true
    wait "${child_pids[@]}" 2>/dev/null || true
  fi
}

trap 'cleanup_children SIGINT; exit 130' INT
trap 'cleanup_children SIGTERM; exit 143' TERM
trap 'cleanup_children EXIT' EXIT

echo "RUN_REPEATS=${RUN_REPEATS}"
echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"

for run_idx in $(seq 1 $RUN_REPEATS); do
  run_output_dir="${BASE_OUTPUT_DIR}_repeat${run_idx}"
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
    --perception-model "${PERCEPTION_MODEL}"
  )

  if [ -n "${PERCEPTION_CONFIG_YAML:-}" ]; then
    cmd+=(--perception-config-yaml "${PERCEPTION_CONFIG_YAML}")
  fi

  if [ -n "${CAPTIONER_CONFIG_YAML:-}" ]; then
    cmd+=(--captioner-config-yaml "${CAPTIONER_CONFIG_YAML}")
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
  "${cmd[@]}" &
  child_pids+=("$!")
done

wait
child_pids=()
trap - INT TERM EXIT
