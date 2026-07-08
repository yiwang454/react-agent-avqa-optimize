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

cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ] && [ -n "${ENV_PREFIX:-}" ] && [ -x "${ENV_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${ENV_PREFIX}/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

# Keep the same starting runtime choices as run_dspy_react_agent_daily_qwen_v6_qwen_seed.sh.
export QWEN_SEED="${QWEN_SEED:-1234}"
export DEEPSEEK_SEED="${DEEPSEEK_SEED:-7}"
export PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa/DSPy/dspy_avqa/yamls/daily_qa_prompt_v6.yaml}"
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa/DSPy/dspy_avqa/yamls/config_localqwen_api_think.yaml}"
export QWEN_BASE_URL="http://10.62.2.181:8000/v1"
export MAX_TURNS="${MIPROV2_MAX_TURNS:-${MAX_TURNS:-4}}"
export PERCEPTION_MODEL="${MIPROV2_PERCEPTION_MODEL:-qwen}"

BASE_ENV_OUTPUT_DIR="${MIPROV2_BASE_ENV_OUTPUT_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_qwenTHINK_v6MULTITURN_miprov2}"
INPUT_JSONL="${MIPROV2_INPUT_JSONL:-${INPUT_JSONL}}"
AUDIO_CAPTION_DIR="${MIPROV2_AUDIO_CAPTION_DIR:-${AUDIO_CAPTION_DIR}}"
BASE_MIPROV2_RUN_DIR="${MIPROV2_RUN_DIR:-${BASE_ENV_OUTPUT_DIR}_qwen_seed_${QWEN_SEED}_deepseek_seed_${DEEPSEEK_SEED}}"
TRAIN_LIMIT="${MIPROV2_TRAIN_LIMIT:-64}"
DEBUG="${MIPROV2_DEBUG:-${DEBUG:-false}}"
DEBUG_LIMIT="${MIPROV2_DEBUG_LIMIT:-${DEBUG_LIMIT:-4}}"

# MIPROv2 optimizer seeds only. QWEN_SEED and DEEPSEEK_SEED stay fixed above.
MIPROV2_SEEDS=(${MIPROV2_SEEDS_OVERRIDE:-0 18 2})
SCRIPT_ARGS=("$@")
child_pids=()

cleanup_children() {
  local reason="${1:-exit}"
  trap - INT TERM EXIT

  if [ "${#child_pids[@]}" -gt 0 ]; then
    echo "Stopping ${#child_pids[@]} background MIPROv2 run(s) after ${reason}..." >&2
    kill -TERM "${child_pids[@]}" 2>/dev/null || true
    wait "${child_pids[@]}" 2>/dev/null || true
  fi
}

trap 'cleanup_children SIGINT; exit 130' INT
trap 'cleanup_children SIGTERM; exit 143' TERM
trap 'cleanup_children EXIT' EXIT

run_miprov2_seed() {
  local miprov2_seed="$1"
  local miprov2_run_dir="${BASE_MIPROV2_RUN_DIR}_miprov2_seed${miprov2_seed}"
  local initial_program="${MIPROV2_INITIAL_PROGRAM:-${miprov2_run_dir}/initial_miprov2_program.json}"
  local output_program="${MIPROV2_OUTPUT_PROGRAM:-${miprov2_run_dir}/compiled_miprov2.json}"
  local metadata_json="${MIPROV2_METADATA_JSON:-${miprov2_run_dir}/compiled_miprov2_metadata.json}"
  local signature_search_json="${MIPROV2_SIGNATURE_SEARCH_JSON:-${miprov2_run_dir}/compiled_miprov2_signature_search.json}"
  local trajectory_jsonl="${MIPROV2_TRAJECTORY_JSONL:-${miprov2_run_dir}/output_test.jsonl}"
  local miprov2_config_file="${miprov2_run_dir}/miprov2_config.json"

  mkdir -p "${miprov2_run_dir}"
  cat > "${miprov2_config_file}" <<JSON
{
  "auto": null,
  "num_candidates": 6,
  "num_trials": 10,
  "max_bootstrapped_demos": 2,
  "max_labeled_demos": 2,
  "seed": ${miprov2_seed},
  "init_temperature": 1.0,
  "num_threads": null,
  "max_errors": null,
  "minibatch": true,
  "minibatch_size": 16,
  "minibatch_full_eval_steps": 5,
  "view_data_batch_size": 8
}
JSON

  echo "============================================================"
  echo "Running DSPy MIPROv2 optimizer: miprov2_seed=${miprov2_seed}"
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
  echo "GEMINI_API_KEY=${GEMINI_API_KEY:+***set***}"
  echo "GEMINI_BASE_URL=${GEMINI_BASE_URL:-}"
  echo "GEMINI_MODEL=${GEMINI_MODEL:-}"
  echo "PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML:-}"
  echo "PROMPT_YAML=${PROMPT_YAML:-}"
  echo "DSPY_AVQA_ALLOWED_TOOLS=${DSPY_AVQA_ALLOWED_TOOLS:-}"
  echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
  echo "INPUT_JSONL=${INPUT_JSONL}"
  echo "AUDIO_CAPTION_DIR=${AUDIO_CAPTION_DIR}"
  echo "BASE_MIPROV2_RUN_DIR=${BASE_MIPROV2_RUN_DIR}"
  echo "MIPROV2_RUN_DIR=${miprov2_run_dir}"
  echo "MIPROV2_CONFIG=${miprov2_config_file}"
  echo "INITIAL_PROGRAM=${initial_program}"
  echo "OUTPUT_PROGRAM=${output_program}"
  echo "METADATA_JSON=${metadata_json}"
  echo "SIGNATURE_SEARCH_JSON=${signature_search_json}"
  echo "TRAJECTORY_JSONL=${trajectory_jsonl}"
  echo "MAX_TURNS=${MAX_TURNS}"
  echo "TRAIN_LIMIT=${TRAIN_LIMIT}"
  echo "DEBUG=${DEBUG}"
  echo "DEBUG_LIMIT=${DEBUG_LIMIT}"

  cmd=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
    --algorithm miprov2
    --input-jsonl "${INPUT_JSONL}"
    --audio-caption-dir "${AUDIO_CAPTION_DIR}"
    --output-program "${output_program}"
    --initial-program "${initial_program}"
    --metadata-json "${metadata_json}"
    --signature-search-json "${signature_search_json}"
    --trajectory-jsonl "${trajectory_jsonl}"
    --max-turns "${MAX_TURNS}"
    --perception-model "${PERCEPTION_MODEL}"
    --miprov2-config "${miprov2_config_file}"
  )

  if [ -n "${TRAIN_LIMIT}" ]; then
    cmd+=(--train-limit "${TRAIN_LIMIT}")
  fi

  if [ -n "${PERCEPTION_CONFIG_YAML:-}" ]; then
    cmd+=(--perception-config-yaml "${PERCEPTION_CONFIG_YAML}")
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

  if [ "${MIPROV2_SKIP_BAD_EXAMPLES:-false}" = "true" ]; then
    cmd+=(--skip-bad-examples)
  fi

  cmd+=("${SCRIPT_ARGS[@]}")
  "${cmd[@]}"
}

echo "MIPROV2_SEEDS=${MIPROV2_SEEDS[*]}"
echo "QWEN_SEED=${QWEN_SEED}"
echo "DEEPSEEK_SEED=${DEEPSEEK_SEED}"

for miprov2_seed in "${MIPROV2_SEEDS[@]}"; do
  run_miprov2_seed "${miprov2_seed}" &
  child_pids+=("$!")
done

wait
child_pids=()
trap - INT TERM EXIT
