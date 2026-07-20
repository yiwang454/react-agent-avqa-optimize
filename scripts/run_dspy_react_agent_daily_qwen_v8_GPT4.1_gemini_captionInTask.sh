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

# ELM GPT planner with Gemini legacy perception and captioning.
export PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE_OVERRIDE:-0.0}"
export PLANNER_OUTPUT_SEQ_LEN="${PLANNER_OUTPUT_SEQ_LEN:-32768}"
export PLANNER_PROVIDER="elm_gpt"
export PLANNER_MODEL="${PLANNER_MODEL_OVERRIDE:-gpt-4.1}"
export PLANNER_REASONING_EFFORT="${PLANNER_REASONING_EFFORT_OVERRIDE:-none}"
export PLANNER_SEED="${PLANNER_SEED_OVERRIDE:-1234}"
export PLANNER_API_KEY="${PLANNER_API_KEY:-${ELM_API_KEY:-}}"
# Preserve the existing planner-key compatibility mapping while ensuring ELM
# mode never inherits a DeepSeek endpoint from the sourced environment file.
export DEEPSEEK_API_KEY="${PLANNER_API_KEY}"
unset DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
if [ -z "${PLANNER_API_KEY}" ]; then
  echo "Set PLANNER_API_KEY or ELM_API_KEY before running." >&2
  exit 2
fi

export PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task.yaml}"
export PERCEPTION_MODEL="${PERCEPTION_MODEL_OVERRIDE:-gemini}"
export CAPTIONER_MODEL="${CAPTIONER_MODEL_OVERRIDE:-gemini}"
GEMINI_RUNTIME_CONFIG_YAML="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${GEMINI_RUNTIME_CONFIG_YAML}}"
export CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML_OVERRIDE:-${GEMINI_RUNTIME_CONFIG_YAML}}"
export DSPY_AVQA_ALLOWED_TOOLS="${DSPY_AVQA_ALLOWED_TOOLS_OVERRIDE:-ask_caption,ask_perception}"
export GEMINI_API_BACKEND="legacy"
export GEMINI_RESPONSE_ERROR_SENSITIVE="${GEMINI_RESPONSE_ERROR_SENSITIVE:-false}"
DEBUG="${DEBUG:-false}"
DEBUG_LIMIT="${DEBUG_LIMIT:-0}"

BASE_ENV_OUTPUT_DIR=/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_v8GeminiCaptionInTask
export OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${BASE_ENV_OUTPUT_DIR}_planner_${PLANNER_MODEL}_seed_${PLANNER_SEED}}"
export OUTPUT_JSONL="${OUTPUT_DIR}/output_test.jsonl"

# Print related settings to verify env loading without leaking secrets.
echo "PLANNER_PROVIDER=${PLANNER_PROVIDER}"
echo "PLANNER_MODEL=${PLANNER_MODEL}"
echo "PLANNER_API_KEY=${PLANNER_API_KEY:+***set***}"
echo "PLANNER_REASONING_EFFORT=${PLANNER_REASONING_EFFORT}"
echo "PLANNER_TEMPERATURE=${PLANNER_TEMPERATURE}"
echo "PLANNER_OUTPUT_SEQ_LEN=${PLANNER_OUTPUT_SEQ_LEN}"
echo "PLANNER_SEED=${PLANNER_SEED}"
echo "GEMINI_API_BACKEND=${GEMINI_API_BACKEND}"
echo "GEMINI_RESPONSE_ERROR_SENSITIVE=${GEMINI_RESPONSE_ERROR_SENSITIVE}"
echo "CAPTIONER_MODEL=${CAPTIONER_MODEL:-}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
echo "PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML:-}"
echo "CAPTIONER_CONFIG_YAML=${CAPTIONER_CONFIG_YAML:-}"
echo "PROMPT_YAML=${PROMPT_YAML:-}"
echo "DSPY_AVQA_ALLOWED_TOOLS=${DSPY_AVQA_ALLOWED_TOOLS:-}"
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "OUTPUT_JSONL=${OUTPUT_JSONL}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "DEBUG=${DEBUG}"
echo "DEBUG_LIMIT=${DEBUG_LIMIT}"

echo "Running DSPy batch runner once."

cmd=(
  python DSPy/avqa_dspy_impl.py
  --input-jsonl "${INPUT_JSONL}"
  --output-jsonl "${OUTPUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --max-turns "${MAX_TURNS}"
  --perception-model "${PERCEPTION_MODEL}"
  --gemini-api-backend legacy
  --signature-in-system-prompt
  --caption-placement task
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
"${cmd[@]}"
