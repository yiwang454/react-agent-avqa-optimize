#!/usr/bin/env bash
set -euo pipefail

# Clean A/B against the GPT-4.1 no-GEPA V8 Gemini baseline: the only runtime
# changes are a four-tool-call budget, the required-perception prompt policy,
# and eight-way inference parallelism.  Captions and perception both remain live
# Gemini calls; no caption cache or GEPA program is used.

PROJECT_DIR="${PROJECT_DIR_OVERRIDE:-.}"
SCRIPT_PATH="./scripts/14_run_dspy_react_agent_daily_gpt4_1_gemini_captionInTask_required_perception_4turn.sh"
ROOT_ENV_FILE="${ROOT_ENV_FILE:-./.env}"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
LOG_FILE="${LOG_FILE_OVERRIDE:-${PROJECT_DIR}/scripts/logs/${SCRIPT_NAME%.sh}.log}"

set -a
source "${ROOT_ENV_FILE}"
set +a

PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX}/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  printf 'Python interpreter is not executable: %s\n' "${PYTHON_BIN}" >&2
  exit 1
fi

PLANNER_CONFIG_YAML="./DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml"
: "${ELM_API_KEY:?ELM_API_KEY must be set in .env}"
export PLANNER_API_KEY="${ELM_API_KEY}"
export DEEPSEEK_API_KEY="${PLANNER_API_KEY}"
unset DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE

export PROMPT_YAML="./DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task_required_perception_4turn.yaml"
export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export PERCEPTION_CONFIG_YAML="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
export CAPTIONER_CONFIG_YAML="${PERCEPTION_CONFIG_YAML}"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="legacy"
export GEMINI_RESPONSE_ERROR_SENSITIVE="false"

MAX_TURNS=5
INFERENCE_NUM_THREADS=8
INFERENCE_BATCH_SIZE=8
export INPUT_JSONL="/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_v8GeminiCaptionInTask_required_perception_maxturns4_threads8_planner_gpt-4.1_seed_1234}"
OUTPUT_JSONL="${OUTPUT_JSONL_OVERRIDE:-${OUTPUT_DIR}/output_test.jsonl}"
ARCHIVED_SCRIPT="${OUTPUT_DIR}/$(basename "${SCRIPT_PATH}")"

if [ -e "${ARCHIVED_SCRIPT}" ] && ! cmp -s "${SCRIPT_PATH}" "${ARCHIVED_SCRIPT}"; then
  printf 'Archived launcher differs from current script: %s\n' "${ARCHIVED_SCRIPT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${LOG_FILE}")"

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

if [ "${DRY_RUN:-false}" = "true" ]; then
  printf 'Dry run: no model request will be sent.\nLog file: %s\nCommand:' "${LOG_FILE}"
  printf ' %q' "${cmd[@]}" "$@"
  printf '\n'
  exit 0
fi

printf 'Log file: %s\nFollow live output with: tail -f %s\n' "${LOG_FILE}" "${LOG_FILE}"
{
  echo "Started: $(date -Is)"
  echo "Planner: elm_gpt/gpt-4.1; captions and perception: live Gemini"
  echo "Prompt: ${PROMPT_YAML}"
  echo "Tool-call budget: ${MAX_TURNS}; required sequence: ask_caption -> ask_perception -> (ask_perception or final)"
  echo "Inference threads/batch size: ${INFERENCE_NUM_THREADS}/${INFERENCE_BATCH_SIZE}"
  echo "Caption cache: disabled"
  echo "Input: ${INPUT_JSONL}"
  echo "Output: ${OUTPUT_JSONL}"
  "${cmd[@]}" "$@"
  echo "Finished: $(date -Is)"
} 2>&1 | tee -a "${LOG_FILE}"

if [ ! -e "${ARCHIVED_SCRIPT}" ]; then
  cp -p "${SCRIPT_PATH}" "${ARCHIVED_SCRIPT}"
fi
