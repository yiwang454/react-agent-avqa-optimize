#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

cd "${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

: "${ELM_API_KEY:?Set ELM_API_KEY for the GPT-4.1 planner.}"
export QWEN_BASE_URL_OVERRIDE="${QWEN_BASE_URL_OVERRIDE:-http://10.62.206.23:8000/v1}"
case "${QWEN_BASE_URL_OVERRIDE}" in
  http://*/v1|https://*/v1) ;;
  *)
    echo "QWEN_BASE_URL_OVERRIDE must be an http(s) URL ending in /v1." >&2
    exit 2
    ;;
esac

export PLANNER_API_KEY="${ELM_API_KEY}"
unset DEEPSEEK_API_KEY DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE

export PERCEPTION_MODEL="qwen"
export CAPTIONER_MODEL="qwen"
export QWEN_BASE_URL_OVERRIDE
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"

PLANNER_CONFIG_YAML="${PLANNER_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/config_localqwen_api_instruct.yaml}"
PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_free_react_caption_in_task.yaml}"
INPUT_JSONL="${INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_caption_cache_v8_qwen3omni_instruct}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1197}"
MAX_TURNS="${MAX_TURNS:-6}"
INFERENCE_NUM_THREADS="${INFERENCE_NUM_THREADS:-4}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-${INFERENCE_NUM_THREADS}}"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
LOG_DIR="${LOG_DIR:-${REPO_DIR}/scripts/logs}"
LOG_FILE="${LOG_FILE_OVERRIDE:-${LOG_DIR}/${SCRIPT_NAME}.log}"

OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_free_react_gpt4_1_qwen3omni_instruct_cached_qwen_caption_maxturns${MAX_TURNS}}"
OUTPUT_JSONL="${OUTPUT_JSONL_OVERRIDE:-${OUTPUT_DIR}/output_test.jsonl}"

for required_file in "${PLANNER_CONFIG_YAML}" "${PERCEPTION_CONFIG_YAML}" "${PROMPT_YAML}" "${INPUT_JSONL}" "${CAPTION_CACHE_DIR}/manifest.json"; do
  if [ ! -f "${required_file}" ]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

actual_count="$(awk 'NF {count++} END {print count+0}' "${INPUT_JSONL}")"
if [ "${actual_count}" -ne "${EXPECTED_COUNT}" ]; then
  echo "Input row count mismatch: expected ${EXPECTED_COUNT}, found ${actual_count} in ${INPUT_JSONL}" >&2
  exit 2
fi
if [ "${MAX_TURNS}" -lt 1 ] || [ "${INFERENCE_NUM_THREADS}" -lt 1 ] || [ "${INFERENCE_BATCH_SIZE}" -lt 1 ]; then
  echo "MAX_TURNS, INFERENCE_NUM_THREADS, and INFERENCE_BATCH_SIZE must be >= 1." >&2
  exit 2
fi

mkdir -p "$(dirname "${LOG_FILE}")"

print_run_info() {
  echo "Planner: elm_gpt/gpt-4.1; perception: qwen3-omni-instruct; captions: validated Qwen cache"
  echo "Prompt: ${PROMPT_YAML}"
  echo "Tool-call budget: ${MAX_TURNS} calls (including the required first ask_caption call)"
  echo "A forced final action, if needed, is outside the tool-call budget."
  echo "Qwen perception base URL: ${QWEN_BASE_URL_OVERRIDE}"
  echo "Caption cache: ${CAPTION_CACHE_DIR}"
  echo "Caption cache scope: all ask_caption calls (engineering-limited reference; live Qwen captions can be brittle or too short)"
  echo "Input: ${INPUT_JSONL} (${actual_count} rows)"
  echo "Output: ${OUTPUT_JSONL}"
  echo "Log file: ${LOG_FILE}"
}

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
  --prompt-yaml "${PROMPT_YAML}"
  --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
  --caption-cache-dir "${CAPTION_CACHE_DIR}"
  --caption-cache-scope all
  --ignore-audio-caption-dir
  --signature-in-system-prompt
  --caption-placement task
  --print-config
)

if [ "${DRY_RUN:-false}" = "true" ]; then
  {
    echo "Dry run: no model request will be sent."
    print_run_info
    printf 'Command:'
    printf ' %q' "${cmd[@]}" "$@"
    printf '\n'
  } | tee -a "${LOG_FILE}"
  exit 0
fi

echo "Log file: ${LOG_FILE}"
echo "Follow live output with: tail -f ${LOG_FILE}"

{
  echo "Started: $(date -Is)"
  print_run_info
  "${cmd[@]}" "$@"
  echo "Finished: $(date -Is)"
} 2>&1 | tee -a "${LOG_FILE}"
