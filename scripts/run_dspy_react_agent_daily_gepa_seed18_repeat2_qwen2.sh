#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_qwen_v2}"

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

# Fixed runtime seeds matching the qwen/deepseek-seed scripts.
export QWEN_SEED="${QWEN_SEED:-1234}"
export DEEPSEEK_SEED="${DEEPSEEK_SEED:-7}"

GEPA_SEED="${GEPA_SEED:-18}"
GEPA_BASE_ENV_OUTPUT_DIR="${GEPA_BASE_ENV_OUTPUT_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_changeoptimize_qwen_v5_gepa_workflow_prompt}"
GEPA_RUN_DIR="${GEPA_RUN_DIR_OVERRIDE:-${GEPA_BASE_ENV_OUTPUT_DIR}_qwen_seed_${QWEN_SEED}_deepseek_seed_${DEEPSEEK_SEED}_gepa_seed${GEPA_SEED}}"
GEPA_COMPILED_PROGRAM="${GEPA_COMPILED_PROGRAM_OVERRIDE:-${GEPA_RUN_DIR}/compiled_gepa.json}"
GEPA_OPTIMIZED_PROMPT_YAML="${GEPA_OPTIMIZED_PROMPT_YAML_OVERRIDE:-${GEPA_RUN_DIR}/planner_workflow_prompt_GEPA_daily_qa_prompt_v5.yaml}"

if [ ! -f "${GEPA_COMPILED_PROGRAM}" ]; then
  echo "Missing GEPA compiled program: ${GEPA_COMPILED_PROGRAM}" >&2
  exit 1
fi

if [ ! -f "${GEPA_OPTIMIZED_PROMPT_YAML}" ]; then
  echo "Missing GEPA optimized prompt YAML: ${GEPA_OPTIMIZED_PROMPT_YAML}" >&2
  exit 1
fi

export PROMPT_YAML="${GEPA_OPTIMIZED_PROMPT_YAML}"
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/config_localqwen_api_instruct.yaml}"
export QWEN_BASE_URL="${QWEN_BASE_URL_OVERRIDE:-http://10.62.141.154:8000/v1}"
export MAX_TURNS="${GEPA_MAX_TURNS:-${MAX_TURNS:-1}}"
export PERCEPTION_MODEL="${GEPA_PERCEPTION_MODEL:-${PERCEPTION_MODEL:-qwen}}"
export DSPY_AVQA_ALLOWED_TOOLS="${DSPY_AVQA_ALLOWED_TOOLS_OVERRIDE:-${DSPY_AVQA_ALLOWED_TOOLS:-ask_perception}}"

INPUT_JSONL="${GEPA_INPUT_JSONL:-${INPUT_JSONL}}"
AUDIO_CAPTION_DIR="${GEPA_AUDIO_CAPTION_DIR:-${AUDIO_CAPTION_DIR}}"
RUN_REPEATS="${RUN_REPEATS:-2}"
DEBUG="${GEPA_DEBUG:-${DEBUG:-false}}"
DEBUG_LIMIT="${GEPA_DEBUG_LIMIT:-${DEBUG_LIMIT:-4}}"

BASE_REPEAT_OUTPUT_DIR="${GEPA_REPEAT_OUTPUT_BASE_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_changeoptimize_qwen_v5_gepa_workflow_prompt_gepa_seed${GEPA_SEED}_rerun}"
export OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${BASE_REPEAT_OUTPUT_DIR}_qwen_seed_${QWEN_SEED}_deepseek_seed_${DEEPSEEK_SEED}}"
BASE_OUTPUT_DIR="${OUTPUT_DIR%/}"

SCRIPT_ARGS=("$@")
child_pids=()

for arg in "${SCRIPT_ARGS[@]}"; do
  case "${arg}" in
    --prompt-yaml|--prompt-yaml=*)
      echo "This script reruns the GEPA seed ${GEPA_SEED} optimized program; do not pass ${arg}." >&2
      exit 2
      ;;
  esac
done

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

echo "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:+***set***}"
echo "DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-}"
echo "DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-}"
echo "DEEPSEEK_SEED=${DEEPSEEK_SEED:-}"
echo "PLANNER_THINKING_MODE=${PLANNER_THINKING_MODE:-}"
echo "PLANNER_TEMPERATURE=${PLANNER_TEMPERATURE:-}"
echo "PLANNER_TOP_P=${PLANNER_TOP_P:-}"
echo "QWEN_API_KEY=${QWEN_API_KEY:+***set***}"
echo "QWEN_BASE_URL=${QWEN_BASE_URL:-}"
echo "QWEN_MODEL=${QWEN_MODEL:-}"
echo "QWEN_ENABLE_THINKING=${QWEN_ENABLE_THINKING:-}"
echo "QWEN_SEED=${QWEN_SEED:-}"
echo "GEPA_SEED=${GEPA_SEED}"
echo "GEPA_RUN_DIR=${GEPA_RUN_DIR}"
echo "GEPA_COMPILED_PROGRAM=${GEPA_COMPILED_PROGRAM}"
echo "GEPA_OPTIMIZED_PROMPT_YAML=${GEPA_OPTIMIZED_PROMPT_YAML}"
echo "PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML:-}"
echo "PROMPT_YAML=${PROMPT_YAML:-}"
echo "DSPY_AVQA_ALLOWED_TOOLS=${DSPY_AVQA_ALLOWED_TOOLS:-}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "AUDIO_CAPTION_DIR=${AUDIO_CAPTION_DIR}"
echo "RUN_REPEATS=${RUN_REPEATS}"
echo "BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "DEBUG=${DEBUG}"
echo "DEBUG_LIMIT=${DEBUG_LIMIT}"

for run_idx in $(seq 1 "${RUN_REPEATS}"); do
  run_output_dir="${BASE_OUTPUT_DIR}_repeat${run_idx}"
  run_output_jsonl="${run_output_dir}/output_test.jsonl"

  echo "Running DSPy batch runner with GEPA seed ${GEPA_SEED} program: run ${run_idx}/${RUN_REPEATS}"
  echo "RUN_OUTPUT_DIR=${run_output_dir}"
  echo "RUN_OUTPUT_JSONL=${run_output_jsonl}"

  cmd=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_impl.py
    --input-jsonl "${INPUT_JSONL}"
    --audio-caption-dir "${AUDIO_CAPTION_DIR}"
    --output-jsonl "${run_output_jsonl}"
    --output-dir "${run_output_dir}"
    --max-turns "${MAX_TURNS}"
    --perception-model "${PERCEPTION_MODEL}"
    --signature-in-system-prompt
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

  cmd+=("${SCRIPT_ARGS[@]}")
  "${cmd[@]}" &
  child_pids+=("$!")
done

wait
child_pids=()
trap - INT TERM EXIT
