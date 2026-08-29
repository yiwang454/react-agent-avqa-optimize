#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="."
SCRIPT_PATH="./scripts/02_run_dspy_react_agent_daily_qwen_v8_GPT_gemini_captionInTask.sh"
ROOT_ENV_FILE="./.env"

set -a
source "${ROOT_ENV_FILE}"
set +a

PLANNER_CONFIG_YAML="./DSPy/dspy_avqa/yamls/reasoner_elm_gpt5_4_none.yaml"
: "${ELM_API_KEY:?ELM_API_KEY must be set in .env}"
export PLANNER_API_KEY="${ELM_API_KEY}"
export DEEPSEEK_API_KEY="${PLANNER_API_KEY}"
unset DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE

export PROMPT_YAML="./DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task.yaml"
export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export PERCEPTION_CONFIG_YAML="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
export CAPTIONER_CONFIG_YAML="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="legacy"
export GEMINI_RESPONSE_ERROR_SENSITIVE="false"
MAX_TURNS=3
DEBUG=false
DEBUG_LIMIT=0
export INPUT_JSONL="/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl"

export OUTPUT_DIR="/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_v8GeminiCaptionInTask_planner_gpt-5.4_seed_1234"
export OUTPUT_JSONL="${OUTPUT_DIR}/output_test.jsonl"
ARCHIVED_SCRIPT="${OUTPUT_DIR}/$(basename "${SCRIPT_PATH}")"
if [ -e "${ARCHIVED_SCRIPT}" ] && ! cmp -s "${SCRIPT_PATH}" "${ARCHIVED_SCRIPT}"; then
  printf 'Archived launcher differs from current script: %s\n' "${ARCHIVED_SCRIPT}" >&2
  exit 2
fi

cmd=(
  python DSPy/avqa_dspy_impl.py
  --input-jsonl "${INPUT_JSONL}"
  --output-jsonl "${OUTPUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --max-turns "${MAX_TURNS}"
  --planner-config-yaml "${PLANNER_CONFIG_YAML}"
  --perception-model "${PERCEPTION_MODEL}"
  --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
  --captioner-config-yaml "${CAPTIONER_CONFIG_YAML}"
  --prompt-yaml "${PROMPT_YAML}"
  --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
  --gemini-api-backend "${GEMINI_API_BACKEND}"
  --signature-in-system-prompt
  --caption-placement task
  --print-config
)
if [ "${DEBUG}" = "true" ]; then
  cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi
cmd+=("$@")
"${cmd[@]}"

if [ ! -e "${ARCHIVED_SCRIPT}" ]; then
  cp -p "${SCRIPT_PATH}" "${ARCHIVED_SCRIPT}"
fi
