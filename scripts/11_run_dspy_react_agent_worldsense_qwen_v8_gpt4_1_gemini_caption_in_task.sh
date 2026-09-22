#!/usr/bin/env bash
set -euo pipefail

# WorldSense inference with the runtime settings of
# 01_run_dspy_react_agent_daily_qwen_v8_GPT4.1_gemini_captionInTask.sh.
# It concurrently runs a 4-thread midshort partition and a 1-thread long partition.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_ENV_FILE="${ROOT_ENV_FILE:-${REPO_DIR}/.env}"
WORLD_SENSE_DIR="${WORLD_SENSE_DIR:-/mnt/ceph_rbd/data/avqa_project/WorldSense}"
MIDSHORT_INPUT_JSONL="${MIDSHORT_INPUT_JSONL:-${WORLD_SENSE_DIR}/worldsense_test_cut_midshort.jsonl}"
LONG_INPUT_JSONL="${LONG_INPUT_JSONL:-${WORLD_SENSE_DIR}/worldsense_test_cut_long.jsonl}"
WORLD_SENSE_VIDEO_DIR="${WORLD_SENSE_VIDEO_DIR:-${WORLD_SENSE_DIR}/videos}"
OUTPUT_DIR="${WORLD_SENSE_OUTPUT_DIR:-${WORLD_SENSE_DIR}/worldsense_dspy_GPT_v8GeminiCaptionInTask_planner_gpt-4.1_seed_1234}"
MIDSHORT_INFERENCE_THREADS="${MIDSHORT_INFERENCE_THREADS:-4}"
LONG_INFERENCE_THREADS="${LONG_INFERENCE_THREADS:-1}"
MAX_TURNS="${MAX_TURNS:-3}"
DEBUG="${DEBUG:-false}"
DEBUG_LIMIT="${DEBUG_LIMIT:-0}"

if [ ! -f "${ROOT_ENV_FILE}" ]; then
  echo "Missing env file: ${ROOT_ENV_FILE}" >&2
  exit 2
fi

set -a
source "${ROOT_ENV_FILE}"
set +a

PYTHON_BIN="${PYTHON_BIN:-${ENV_PREFIX:-}/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
: "${ELM_API_KEY:?ELM_API_KEY must be set in ${ROOT_ENV_FILE}}"

export PROJECT_DIR="${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"
export PLANNER_API_KEY="${ELM_API_KEY}"
export DEEPSEEK_API_KEY="${PLANNER_API_KEY}"
unset DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="legacy"
export GEMINI_RESPONSE_ERROR_SENSITIVE="false"

PLANNER_CONFIG_YAML="${PLANNER_CONFIG_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
PROMPT_YAML="${PROMPT_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task.yaml}"
GEMINI_RUNTIME_CONFIG_YAML="${GEMINI_RUNTIME_CONFIG_YAML:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml}"
PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML:-${GEMINI_RUNTIME_CONFIG_YAML}}"
CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML:-${GEMINI_RUNTIME_CONFIG_YAML}}"

for required_path in "${PLANNER_CONFIG_YAML}" "${PROMPT_YAML}" "${PERCEPTION_CONFIG_YAML}" "${CAPTIONER_CONFIG_YAML}"; do
  if [ ! -f "${required_path}" ]; then
    echo "Missing required configuration: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
SCRIPT_ARGS=("$@")
child_pids=()

prepare_partition_input() {
  local partition="$1"
  local input_jsonl="$2"
  local partition_dir="${OUTPUT_DIR}/${partition}"
  mkdir -p "${partition_dir}"
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_worldsense_local_cut_jsonl.py" \
    --source-jsonl "${input_jsonl}" \
    --video-dir "${WORLD_SENSE_VIDEO_DIR}" \
    --output-jsonl "${partition_dir}/worldsense_test_cut.local_paths.jsonl"
}

run_partition() {
  local partition="$1"
  local inference_threads="$2"
  local partition_dir="${OUTPUT_DIR}/${partition}"
  local local_input_jsonl="${partition_dir}/worldsense_test_cut.local_paths.jsonl"
  local output_jsonl="${partition_dir}/output_test.jsonl"
  local log_file="${partition_dir}/run.log"
  local -a cmd=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_impl.py
    --input-jsonl "${local_input_jsonl}"
    --output-jsonl "${output_jsonl}"
    --output-dir "${partition_dir}"
    --inference-num-threads "${inference_threads}"
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

  echo "${partition}: inference threads=${inference_threads}"
  echo "${partition}: log file=${log_file}"
  echo "${partition}: output JSONL=${output_jsonl}"
  {
    echo "partition=${partition}"
    echo "input_jsonl=${local_input_jsonl}"
    echo "inference_num_threads=${inference_threads}"
    echo "output_jsonl=${output_jsonl}"
  } >> "${log_file}"
  "${cmd[@]}" "${SCRIPT_ARGS[@]}" >> "${log_file}" 2>&1 &
  child_pids+=("$!")
}

cleanup_children() {
  trap - INT TERM EXIT
  if [ "${#child_pids[@]}" -gt 0 ]; then
    kill -TERM "${child_pids[@]}" 2>/dev/null || true
    wait "${child_pids[@]}" 2>/dev/null || true
  fi
}

prepare_partition_input midshort "${MIDSHORT_INPUT_JSONL}"
prepare_partition_input long "${LONG_INPUT_JSONL}"

cd "${PROJECT_DIR}"
trap 'cleanup_children; exit 130' INT
trap 'cleanup_children; exit 143' TERM
trap 'cleanup_children' EXIT
run_partition midshort "${MIDSHORT_INFERENCE_THREADS}"
run_partition long "${LONG_INFERENCE_THREADS}"
wait "${child_pids[@]}"
child_pids=()
trap - INT TERM EXIT
