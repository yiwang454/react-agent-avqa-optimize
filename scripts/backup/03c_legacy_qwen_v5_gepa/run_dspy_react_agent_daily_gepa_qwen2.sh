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

# Keep the same starting runtime choices as run_dspy_react_agent_daily_qwen_v5_qwen_seed.sh.
export QWEN_SEED="${QWEN_SEED:-1234}"
export DEEPSEEK_SEED="${DEEPSEEK_SEED:-7}"
export PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v5.yaml}"
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/config_localqwen_api_instruct.yaml}"
export QWEN_BASE_URL="${QWEN_BASE_URL_OVERRIDE:-http://10.62.141.154:8000/v1}"
export MAX_TURNS="${GEPA_MAX_TURNS:-${MAX_TURNS:-4}}"
export PERCEPTION_MODEL="${GEPA_PERCEPTION_MODEL:-qwen}"

OPTIMIZE_TARGET="planner.workflow_prompt"
START_PROMPT_YAML_NAME="${GEPA_START_PROMPT_YAML_NAME:-$(basename "${PROMPT_YAML}")}"
START_PROMPT_YAML_NAME="${START_PROMPT_YAML_NAME%.*}"
SIGNATURE_IN_SYSTEM_PROMPT="${GEPA_SIGNATURE_IN_SYSTEM_PROMPT:-true}"

BASE_ENV_OUTPUT_DIR="${GEPA_BASE_ENV_OUTPUT_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_changeoptimize_qwen_v5_gepa_workflow_prompt}"
INPUT_JSONL="${GEPA_INPUT_JSONL:-${INPUT_JSONL}}"
TRAINSET_JSONL="${GEPA_TRAINSET_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedTrain125.jsonl}"
VALSET_JSONL="${GEPA_VALSET_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedVal125.jsonl}"
AUDIO_CAPTION_DIR="${GEPA_AUDIO_CAPTION_DIR:-${AUDIO_CAPTION_DIR}}"
BASE_GEPA_RUN_DIR="${GEPA_RUN_DIR:-${BASE_ENV_OUTPUT_DIR}_qwen_seed_${QWEN_SEED}_deepseek_seed_${DEEPSEEK_SEED}}"
TRAIN_LIMIT="${GEPA_TRAIN_LIMIT:-64}"
DEBUG="${GEPA_DEBUG:-${DEBUG:-false}}"
DEBUG_LIMIT="${GEPA_DEBUG_LIMIT:-${DEBUG_LIMIT:-4}}"

# GEPA optimizer seeds only. QWEN_SEED and DEEPSEEK_SEED stay fixed above.
GEPA_SEEDS=(${GEPA_SEEDS_OVERRIDE:-0 18 2})
SCRIPT_ARGS=("$@")
child_pids=()

for arg in "${SCRIPT_ARGS[@]}"; do
  case "${arg}" in
    --optimize-target|--optimize-target=*)
      echo "This script optimizes only ${OPTIMIZE_TARGET}; do not pass ${arg}." >&2
      exit 2
      ;;
  esac
done

cleanup_children() {
  local reason="${1:-exit}"
  trap - INT TERM EXIT

  if [ "${#child_pids[@]}" -gt 0 ]; then
    echo "Stopping ${#child_pids[@]} background GEPA run(s) after ${reason}..." >&2
    kill -TERM "${child_pids[@]}" 2>/dev/null || true
    wait "${child_pids[@]}" 2>/dev/null || true
  fi
}

trap 'cleanup_children SIGINT; exit 130' INT
trap 'cleanup_children SIGTERM; exit 143' TERM
trap 'cleanup_children EXIT' EXIT

run_gepa_seed() {
  local gepa_seed="$1"
  local gepa_run_dir="${BASE_GEPA_RUN_DIR}_gepa_seed${gepa_seed}"
  local initial_program="${gepa_run_dir}/initial_gepa_program.json"
  local output_program="${gepa_run_dir}/compiled_gepa.json"
  local metadata_json="${gepa_run_dir}/compiled_gepa_metadata.json"
  local signature_search_json="${gepa_run_dir}/compiled_gepa_signature_search.json"
  local trajectory_jsonl="${GEPA_TRAJECTORY_JSONL:-${gepa_run_dir}/optimized_trainset_trajectories.jsonl}"
  local final_eval_output_jsonl="${GEPA_FINAL_EVAL_OUTPUT_JSONL:-${gepa_run_dir}/output_test.jsonl}"
  local final_eval_output_dir="${GEPA_FINAL_EVAL_OUTPUT_DIR:-${gepa_run_dir}}"
  local optimizer_log_dir="${GEPA_OPTIMIZER_LOG_DIR:-${gepa_run_dir}/optimizer_logs}"
  local gepa_config_file="${gepa_run_dir}/gepa_config.json"
  local gepa_log_dir="${gepa_run_dir}/gepa_logs"
  local safe_optimize_target="${OPTIMIZE_TARGET//./_}"
  local optimized_prompt_config_yaml="${GEPA_OPTIMIZED_PROMPT_CONFIG_YAML:-${gepa_run_dir}/${safe_optimize_target}_GEPA_${START_PROMPT_YAML_NAME}.yaml}"

  mkdir -p "${gepa_run_dir}"
  cat > "${gepa_config_file}" <<JSON
{
  "auto": null,
  "max_full_evals": 6,
  "max_metric_calls": null,
  "reflection_minibatch_size": 3,
  "candidate_selection_strategy": "pareto",
  "skip_perfect_score": true,
  "use_merge": true,
  "max_merge_invocations": 5,
  "num_threads": null,
  "seed": ${gepa_seed},
  "log_dir": "${gepa_log_dir}",
  "track_stats": true,
  "track_best_outputs": true
}
JSON

  echo "============================================================"
  echo "Running DSPy GEPA optimizer: gepa_seed=${gepa_seed}"
  echo "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:+***set***}"
  echo "DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-}"
  echo "DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-}"
  echo "DEEPSEEK_SEED=${DEEPSEEK_SEED:-}"
  echo "PLANNER_THINKING_MODE=${PLANNER_THINKING_MODE:-}"
  echo "PLANNER_TEMPERATURE=${PLANNER_TEMPERATURE:-}"
  echo "GEPA_REFLECTION_TEMPERATURE=${GEPA_REFLECTION_TEMPERATURE:-<same as planner>}"
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
  echo "OPTIMIZE_TARGET=${OPTIMIZE_TARGET}"
  echo "START_PROMPT_YAML_NAME=${START_PROMPT_YAML_NAME}"
  echo "OPTIMIZED_PROMPT_CONFIG_YAML=${optimized_prompt_config_yaml}"
  echo "SIGNATURE_IN_SYSTEM_PROMPT=${SIGNATURE_IN_SYSTEM_PROMPT}"
  echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
  echo "INPUT_JSONL=${INPUT_JSONL}"
  echo "TRAINSET_JSONL=${TRAINSET_JSONL}"
  echo "VALSET_JSONL=${VALSET_JSONL}"
  echo "AUDIO_CAPTION_DIR=${AUDIO_CAPTION_DIR}"
  echo "BASE_GEPA_RUN_DIR=${BASE_GEPA_RUN_DIR}"
  echo "GEPA_RUN_DIR=${gepa_run_dir}"
  echo "GEPA_CONFIG=${gepa_config_file}"
  echo "INITIAL_PROGRAM=${initial_program}"
  echo "OUTPUT_PROGRAM=${output_program}"
  echo "METADATA_JSON=${metadata_json}"
  echo "SIGNATURE_SEARCH_JSON=${signature_search_json}"
  echo "TRAJECTORY_JSONL=${trajectory_jsonl}"
  echo "FINAL_EVAL_OUTPUT_JSONL=${final_eval_output_jsonl}"
  echo "FINAL_EVAL_OUTPUT_DIR=${final_eval_output_dir}"
  echo "OPTIMIZER_LOG_DIR=${optimizer_log_dir}"
  echo "MAX_TURNS=${MAX_TURNS}"
  echo "TRAIN_LIMIT=${TRAIN_LIMIT}"
  echo "DEBUG=${DEBUG}"
  echo "DEBUG_LIMIT=${DEBUG_LIMIT}"

  cmd=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
    --algorithm gepa
    --optimize-target "${OPTIMIZE_TARGET}"
    --input-jsonl "${INPUT_JSONL}"
    --trainset-jsonl "${TRAINSET_JSONL}"
    --valset-jsonl "${VALSET_JSONL}"
    --audio-caption-dir "${AUDIO_CAPTION_DIR}"
    --output-program "${output_program}"
    --initial-program "${initial_program}"
    --metadata-json "${metadata_json}"
    --signature-search-json "${signature_search_json}"
    --trajectory-jsonl "${trajectory_jsonl}"
    --optimizer-log-dir "${optimizer_log_dir}"
    --optimized-prompt-config-yaml "${optimized_prompt_config_yaml}"
    --final-eval-output-jsonl "${final_eval_output_jsonl}"
    --final-eval-output-dir "${final_eval_output_dir}"
    --max-turns "${MAX_TURNS}"
    --perception-model "${PERCEPTION_MODEL}"
    --gepa-config "${gepa_config_file}"
    --signature-in-system-prompt
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

  # if [ "${SIGNATURE_IN_SYSTEM_PROMPT}" = "true" ]; then
  #   cmd+=(--signature-in-system-prompt)
  # fi

  if [ "${DEBUG}" = "true" ]; then
    cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
  fi

  if [ "${GEPA_SKIP_BAD_EXAMPLES:-false}" = "true" ]; then
    cmd+=(--skip-bad-examples)
  fi

  cmd+=("${SCRIPT_ARGS[@]}")
  "${cmd[@]}"
}

echo "GEPA_SEEDS=${GEPA_SEEDS[*]}"
echo "QWEN_SEED=${QWEN_SEED}"
echo "DEEPSEEK_SEED=${DEEPSEEK_SEED}"

for gepa_seed in "${GEPA_SEEDS[@]}"; do
  run_gepa_seed "${gepa_seed}" &
  child_pids+=("$!")
done

wait
child_pids=()
trap - INT TERM EXIT
