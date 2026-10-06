#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="."
SCRIPT_PATH="./scripts/14_run_dspy_react_agent_daily_gpt4_1_gemini_captionInTask_g0_simplified.sh"
ROOT_ENV_FILE="./.env"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
LOG_FILE="${PROJECT_DIR}/scripts/logs/${SCRIPT_NAME%.sh}.log"

set -a
source "${ROOT_ENV_FILE}"
set +a

PYTHON_BIN="${ENV_PREFIX}/bin/python"
if [ ! -x "${PYTHON_BIN}" ]; then
  printf 'Python interpreter is not executable: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi

PLANNER_CONFIG_YAML="./DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml"
: "${ELM_API_KEY:?ELM_API_KEY must be set in .env}"
export PLANNER_API_KEY="${ELM_API_KEY}"
export DEEPSEEK_API_KEY="${PLANNER_API_KEY}"
unset DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE

export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export PERCEPTION_CONFIG_YAML="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
export CAPTIONER_CONFIG_YAML="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="legacy"
export GEMINI_RESPONSE_ERROR_SENSITIVE="false"
MAX_TURNS=3
INFERENCE_NUM_THREADS=6
INFERENCE_BATCH_SIZE=6
DEBUG=false
DEBUG_LIMIT=0
export INPUT_JSONL="/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl"

echo "${LOG_FILE}"

PROMPT_YAMLS=(
  "./DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task_gpt4_1_g0_simplified.yaml"
  "./DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task_gpt4_1_g0_quatered.yaml"
)
OUTPUT_DIRS=(
  "/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_v8GeminiCaptionInTask_g0Simplified_planner_gpt-4.1_seed_1234_repeat2"
  "/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_v8GeminiCaptionInTask_g0Quatered_planner_gpt-4.1_seed_1234_repeat1"
)

for RUN_INDEX in "${!PROMPT_YAMLS[@]}"; do
  export PROMPT_YAML="${PROMPT_YAMLS[RUN_INDEX]}"
  export OUTPUT_DIR="${OUTPUT_DIRS[RUN_INDEX]}"
  export OUTPUT_JSONL="${OUTPUT_DIR}/output_test.jsonl"
  ARCHIVED_SCRIPT="${OUTPUT_DIR}/$(basename "${SCRIPT_PATH}")"
  if [ -e "${ARCHIVED_SCRIPT}" ] && ! cmp -s "${SCRIPT_PATH}" "${ARCHIVED_SCRIPT}"; then
    printf 'Archived launcher differs from current script: %s\n' "${ARCHIVED_SCRIPT}" >&2
    exit 2
  fi

  cmd=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_impl.py
    --input-jsonl "${INPUT_JSONL}"
    --output-jsonl "${OUTPUT_JSONL}"
    --output-dir "${OUTPUT_DIR}"
    --max-turns "${MAX_TURNS}"
    --inference-num-threads "${INFERENCE_NUM_THREADS}"
    --inference-batch-size "${INFERENCE_BATCH_SIZE}"
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
  "${cmd[@]}" >> "${LOG_FILE}" 2>&1

  if [ ! -e "${ARCHIVED_SCRIPT}" ]; then
    cp -p "${SCRIPT_PATH}" "${ARCHIVED_SCRIPT}"
  fi
done
