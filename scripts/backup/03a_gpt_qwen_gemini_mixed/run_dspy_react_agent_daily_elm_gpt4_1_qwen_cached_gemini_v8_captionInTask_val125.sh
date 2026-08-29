#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_qwen_v8}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

export PROJECT_DIR="${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ] && [ -n "${ENV_PREFIX:-}" ] && [ -x "${ENV_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${ENV_PREFIX}/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

# Match the GEPA runtime: GPT-4.1 planner, Qwen perception, and cached Gemini captions.
export PLANNER_PROVIDER="elm_gpt"
export PLANNER_MODEL="${PLANNER_MODEL_OVERRIDE:-gpt-4.1}"
export PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE_OVERRIDE:-0.0}"
export PLANNER_OUTPUT_SEQ_LEN="${PLANNER_OUTPUT_SEQ_LEN:-32768}"
export PLANNER_TOP_P="${PLANNER_TOP_P_OVERRIDE:-1.0}"
export PLANNER_REASONING_EFFORT="${PLANNER_REASONING_EFFORT_OVERRIDE:-none}"
export PLANNER_SEED="${PLANNER_SEED_OVERRIDE:-1234}"
export PLANNER_API_KEY="${PLANNER_API_KEY:-${ELM_API_KEY:-}}"
unset DEEPSEEK_API_KEY DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
if [ -z "${PLANNER_API_KEY}" ]; then
  echo "Set PLANNER_API_KEY or ELM_API_KEY before running." >&2
  exit 2
fi

export QWEN_SEED="${QWEN_SEED:-1234}"
export PERCEPTION_MODEL="qwen"
export CAPTIONER_MODEL="gemini"
export PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task.yaml}"
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/config_localqwen_api_instruct.yaml}"
export DSPY_AVQA_ALLOWED_TOOLS="${DSPY_AVQA_ALLOWED_TOOLS_OVERRIDE:-ask_caption,ask_perception}"
export QWEN_BASE_URL="${QWEN_BASE_URL_OVERRIDE:-http://xxx:8000/v1}"

VALID125_JSONL="${VALID125_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedVal125.jsonl}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_caption_cache_v8_gemini_from3repeats}"
INFERENCE_NUM_THREADS="${INFERENCE_NUM_THREADS:-${GEPA_NUM_THREADS:-4}}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-${INFERENCE_NUM_THREADS}}"
MAX_TURNS="${MAX_TURNS:-3}"
DEBUG="${DEBUG:-false}"
DEBUG_LIMIT="${DEBUG_LIMIT:-4}"

if [ ! -f "${VALID125_JSONL}" ]; then
  echo "Valid125 input JSONL not found: ${VALID125_JSONL}" >&2
  exit 1
fi
if [ ! -f "${CAPTION_CACHE_DIR}/manifest.json" ]; then
  echo "Caption cache manifest not found: ${CAPTION_CACHE_DIR}/manifest.json" >&2
  exit 1
fi
if [ "${INFERENCE_NUM_THREADS}" -lt 1 ] || [ "${INFERENCE_BATCH_SIZE}" -lt 1 ]; then
  echo "INFERENCE_NUM_THREADS and INFERENCE_BATCH_SIZE must both be >= 1." >&2
  exit 2
fi

BASE_OUTPUT_DIR="/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_qwen_cached_gemini_v8_captionInTask_val125"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${BASE_OUTPUT_DIR}_planner_${PLANNER_MODEL}_planner_seed_${PLANNER_SEED}_qwen_seed_${QWEN_SEED}}"
OUTPUT_JSONL="${OUTPUT_DIR}/output_test.jsonl"

echo "Planner=${PLANNER_PROVIDER}/${PLANNER_MODEL}; reasoning_effort=${PLANNER_REASONING_EFFORT}; planner_key=${PLANNER_API_KEY:+***set***}"
echo "Qwen perception base url=${QWEN_BASE_URL}; qwen_seed=${QWEN_SEED}"
echo "Caption cache=${CAPTION_CACHE_DIR}"
echo "Valid125 input=${VALID125_JSONL}"
echo "Inference threads=${INFERENCE_NUM_THREADS}; batch_size=${INFERENCE_BATCH_SIZE}"
echo "Output=${OUTPUT_JSONL}"

cmd=(
  "${PYTHON_BIN}" DSPy/avqa_dspy_impl.py
  --input-jsonl "${VALID125_JSONL}"
  --output-jsonl "${OUTPUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --max-turns "${MAX_TURNS}"
  --inference-num-threads "${INFERENCE_NUM_THREADS}"
  --inference-batch-size "${INFERENCE_BATCH_SIZE}"
  --perception-model "${PERCEPTION_MODEL}"
  --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
  --prompt-yaml "${PROMPT_YAML}"
  --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
  --caption-cache-dir "${CAPTION_CACHE_DIR}"
  --ignore-audio-caption-dir
  --signature-in-system-prompt
  --caption-placement task
)

if [ "${DEBUG}" = "true" ]; then
  cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi

"${cmd[@]}" "$@"
