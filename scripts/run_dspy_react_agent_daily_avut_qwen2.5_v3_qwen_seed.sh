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

export QWEN_SEED="${QWEN_SEED:-1234}"
export DEEPSEEK_SEED="${DEEPSEEK_SEED:-7}"
export MAX_TURNS="${MAX_TURNS:-4}"
export DEBUG="${DEBUG:-false}"
export DEBUG_LIMIT="${DEBUG_LIMIT:-4}"
export PERCEPTION_MODEL="${PERCEPTION_MODEL:-qwen}"

# Difference from the qwen3omni v3 script:
# use the qwen2.5 perception config requested for this experiment.
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/config_localqwen2.5_api.yaml}"

# Difference from the qwen3omni v3 script:
# qwen2.5 local API works with file media URLs and does not include the qwen3
# enable_thinking request field by default.
export QWEN_MEDIA_URL_MODE="http://10.62.231.10:8000/v1"
export QWEN_STREAM="${QWEN_STREAM:-0}"
export QWEN_INCLUDE_ENABLE_THINKING="${QWEN_INCLUDE_ENABLE_THINKING:-0}"

# Difference from the qwen3omni v3 script:
# do not hard-code the qwen3omni endpoint. Let the qwen2.5 YAML set it, unless
# QWEN_BASE_URL_OVERRIDE is provided explicitly.
if [ -n "${QWEN_BASE_URL_OVERRIDE:-}" ]; then
  export QWEN_BASE_URL="${QWEN_BASE_URL_OVERRIDE}"
else
  unset QWEN_BASE_URL
fi

DAILY_PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa/DSPy/dspy_avqa/yamls/daily_qa_prompt_v3_qwen2.5.yaml}"

# Difference from the qwen3omni v3 script:
# AVUT is added after daily omni. The baseline qwen2.5 script switches from
# config_daily_qa_prompt_repetition.yaml to config_AVUT_qa_prompt_repetition.yaml;
# DSPy prompt YAMLs use a different planner/perception schema, so AVUT keeps the
# v3 DSPy prompt unless AVUT_PROMPT_YAML_OVERRIDE is set.
AVUT_PROMPT_YAML="${AVUT_PROMPT_YAML_OVERRIDE:-${DAILY_PROMPT_YAML}}"

# Difference from the qwen3omni v3 script:
# use the same daily/AVUT input-jsonl split as the qwen2.5 baseline reference.
DAILY_INPUT_JSONL="${DAILY_INPUT_JSONL_OVERRIDE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl}"
AVUT_INPUT_JSONL="${AVUT_INPUT_JSONL_OVERRIDE:-/mnt/ceph_rbd/data/avqa_project/avut_jsonls/avut_cuts_human_av_seperated.jsonl}"

DAILY_AUDIO_CAPTION_DIR="${DAILY_AUDIO_CAPTION_DIR_OVERRIDE:-${AUDIO_CAPTION_DIR:-}}"
# DSPy requires response JSONs keyed by AVUT cut id. Override this with a true
# AVUT caption directory if one is available.
AVUT_AUDIO_CAPTION_DIR="${AVUT_AUDIO_CAPTION_DIR_OVERRIDE:-${AVUT_AUDIO_CAPTION_DIR:-/mnt/ceph_rbd/data/avqa_project/AVUTBenchmark/results/AVUT_think_local_qwen3omni}}"

# Difference from the qwen3omni v3 script:
# output directories explicitly include qwen2.5 and are split by dataset.
DAILY_BASE_OUTPUT_DIR="${DAILY_OUTPUT_DIR_OVERRIDE:-${OUTPUT_DIR_OVERRIDE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_qwen2.5_v3SYS_qwen_seed_${QWEN_SEED}_deepseek_seed_${DEEPSEEK_SEED}}}"
AVUT_BASE_OUTPUT_DIR="${AVUT_OUTPUT_DIR_OVERRIDE:-/mnt/ceph_rbd/data/avqa_project/AVUTBenchmark/results/AVUT_dspy_qwen2.5_v3SYS_qwen_seed_${QWEN_SEED}_deepseek_seed_${DEEPSEEK_SEED}}"

RUN_REPEATS="${RUN_REPEATS:-3}"
# Difference from the original qwen3omni v3 script:
# this fresh qwen2.5 run starts at repeat1 by default; set RUN_START_IDX=2 to
# mimic the old v3 script's resume behavior.
RUN_START_IDX="${RUN_START_IDX:-1}"

run_dataset() {
  local dataset_name="$1"
  local input_jsonl="$2"
  local audio_caption_dir="$3"
  local prompt_yaml="$4"
  local base_output_dir="${5%/}"
  shift 5
  local child_pids=()

  cleanup_children() {
    local reason="${1:-exit}"
    trap - INT TERM EXIT

    if [ "${#child_pids[@]}" -gt 0 ]; then
      echo "Stopping ${#child_pids[@]} background run(s) for ${dataset_name} after ${reason}..." >&2
      kill -TERM "${child_pids[@]}" 2>/dev/null || true
      wait "${child_pids[@]}" 2>/dev/null || true
    fi
  }

  trap 'cleanup_children SIGINT; exit 130' INT
  trap 'cleanup_children SIGTERM; exit 143' TERM
  trap 'cleanup_children EXIT' EXIT

  if [ -z "${audio_caption_dir}" ]; then
    echo "Missing audio caption directory for ${dataset_name}" >&2
    exit 1
  fi

  echo "=== ${dataset_name} ==="
  echo "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:+***set***}"
  echo "DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-}"
  echo "DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-}"
  echo "DEEPSEEK_SEED=${DEEPSEEK_SEED}"
  echo "PLANNER_THINKING_MODE=${PLANNER_THINKING_MODE:-}"
  echo "PLANNER_TEMPERATURE=${PLANNER_TEMPERATURE:-}"
  echo "PLANNER_TOP_P=${PLANNER_TOP_P:-}"
  echo "QWEN_API_KEY=${QWEN_API_KEY:+***set***}"
  echo "QWEN_BASE_URL=${QWEN_BASE_URL:-<from ${PERCEPTION_CONFIG_YAML}>}"
  echo "QWEN_MODEL=${QWEN_MODEL:-<from ${PERCEPTION_CONFIG_YAML}>}"
  echo "QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-<from ${PERCEPTION_CONFIG_YAML}>}"
  echo "QWEN_MEDIA_URL_MODE=${QWEN_MEDIA_URL_MODE}"
  echo "QWEN_STREAM=${QWEN_STREAM}"
  echo "QWEN_INCLUDE_ENABLE_THINKING=${QWEN_INCLUDE_ENABLE_THINKING}"
  echo "QWEN_SEED=${QWEN_SEED}"
  echo "PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML}"
  echo "PROMPT_YAML=${prompt_yaml}"
  echo "DSPY_AVQA_ALLOWED_TOOLS=${DSPY_AVQA_ALLOWED_TOOLS:-}"
  echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
  echo "INPUT_JSONL=${input_jsonl}"
  echo "AUDIO_CAPTION_DIR=${audio_caption_dir}"
  echo "BASE_OUTPUT_DIR=${base_output_dir}"
  echo "MAX_TURNS=${MAX_TURNS}"
  echo "DEBUG=${DEBUG}"
  echo "DEBUG_LIMIT=${DEBUG_LIMIT}"
  echo "RUN_REPEATS=${RUN_REPEATS}"
  echo "RUN_START_IDX=${RUN_START_IDX}"

  for run_idx in $(seq "${RUN_START_IDX}" "${RUN_REPEATS}"); do
    local run_output_dir="${base_output_dir}_repeat${run_idx}"
    local run_output_jsonl="${run_output_dir}/output_test.jsonl"

    echo "Running DSPy batch runner: ${dataset_name} run ${run_idx}/${RUN_REPEATS}"
    echo "RUN_OUTPUT_DIR=${run_output_dir}"
    echo "RUN_OUTPUT_JSONL=${run_output_jsonl}"

    cmd=(
      python DSPy/avqa_dspy_impl.py
      --input-jsonl "${input_jsonl}"
      --audio-caption-dir "${audio_caption_dir}"
      --output-jsonl "${run_output_jsonl}"
      --output-dir "${run_output_dir}"
      --max-turns "${MAX_TURNS}"
      --perception-model "${PERCEPTION_MODEL}"
      --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
      --prompt-yaml "${prompt_yaml}"
    )

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
}

run_dataset "daily_omni" "${DAILY_INPUT_JSONL}" "${DAILY_AUDIO_CAPTION_DIR}" "${DAILY_PROMPT_YAML}" "${DAILY_BASE_OUTPUT_DIR}" "$@"
# run_dataset "AVUT" "${AVUT_INPUT_JSONL}" "${AVUT_AUDIO_CAPTION_DIR}" "${AVUT_PROMPT_YAML}" "${AVUT_BASE_OUTPUT_DIR}" "$@"
