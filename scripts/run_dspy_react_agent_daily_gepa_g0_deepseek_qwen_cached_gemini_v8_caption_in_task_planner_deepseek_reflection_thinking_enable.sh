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

export GEPA_EXPERIMENT_NAME="${GEPA_EXPERIMENT_NAME:-g0_deepseek_qwen_cached_gemini_v8_caption_in_task_planner_reflectionDeepseek}"
export OPTIMIZE_TARGETS_CSV="planner.workflow_prompt"
export PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task.yaml}"
export GEPA_OPTIMIZED_PROMPT_CONFIG_BASENAME="daily_qa_prompt_v8_caption_in_task.yaml"

export PLANNER_PROVIDER="deepseek"
export DEEPSEEK_MODEL="${DEEPSEEK_MODEL_OVERRIDE:-${DEEPSEEK_MODEL:-deepseek-v4-pro}}"
export PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE_OVERRIDE:-0.0}"
export PLANNER_TOP_P="${PLANNER_TOP_P_OVERRIDE:-1.0}"
export PLANNER_THINKING_MODE="${PLANNER_THINKING_MODE_OVERRIDE:-disabled}"
export DEEPSEEK_SEED="${DEEPSEEK_SEED_OVERRIDE:-7}"
unset PLANNER_REASONING_EFFORT PLANNER_SEED

if [ -z "${DEEPSEEK_API_KEY:-${DEEPSEEK_TOKEN:-}}" ]; then
  echo "Set DEEPSEEK_API_KEY or DEEPSEEK_TOKEN before running." >&2
  exit 2
fi

export QWEN_SEED="${QWEN_SEED:-1234}"
export PERCEPTION_MODEL="qwen"
export CAPTIONER_MODEL="gemini"
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/config_localqwen_api_instruct.yaml}"
export CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml}"
export DSPY_AVQA_ALLOWED_TOOLS="${DSPY_AVQA_ALLOWED_TOOLS_OVERRIDE:-ask_caption,ask_perception}"
export QWEN_BASE_URL="${QWEN_BASE_URL_OVERRIDE:-http://10.62.186.38:8000/v1}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_caption_cache_v8_gemini_from3repeats}"
if [ ! -f "${CAPTION_CACHE_DIR}/manifest.json" ]; then
  echo "Caption cache manifest not found: ${CAPTION_CACHE_DIR}/manifest.json" >&2
  exit 1
fi

export GEPA_CAPTION_SUPERVISION="none"
export GEPA_REFLECTION_TEMPLATE_VERSION="${GEPA_REFLECTION_TEMPLATE_VERSION:-original}"
export GEPA_REFLECTION_MODEL="${GEPA_REFLECTION_MODEL_OVERRIDE:-${DEEPSEEK_MODEL}}"
# A distinct reflection model otherwise drops an inherited DeepSeek endpoint.
export GEPA_REFLECTION_API_KEY="${GEPA_REFLECTION_API_KEY:-${DEEPSEEK_API_KEY:-${DEEPSEEK_TOKEN:-}}}"
export GEPA_REFLECTION_API_BASE="${GEPA_REFLECTION_API_BASE:-${DEEPSEEK_BASE_URL:-https://api.deepseek.com}}"
export GEPA_REFLECTION_THINKING_MODE="${GEPA_REFLECTION_THINKING_MODE_OVERRIDE:-enable}"
unset GEPA_REFLECTION_REASONING_EFFORT GEPA_REFLECTION_TEMPERATURE

BASE_ENV_OUTPUT_DIR="${GEPA_BASE_ENV_OUTPUT_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_deepseek_qwen_cached_gemini_v8_gepa_${GEPA_EXPERIMENT_NAME}}"
INPUT_JSONL="${GEPA_INPUT_JSONL:-${INPUT_JSONL}}"
DAILY_OMNI_ROOT="${GEPA_DAILY_OMNI_ROOT:-${DAILY_OMNI_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}}"
TRAINSET_JSONL="${GEPA_TRAINSET_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedTrain125.jsonl}"
VALSET_JSONL="${GEPA_VALSET_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedVal125.jsonl}"
TRAIN_LIMIT="${GEPA_TRAIN_LIMIT:-64}"
MAX_TURNS="${GEPA_MAX_TURNS:-${MAX_TURNS:-3}}"
GEPA_MAX_FULL_EVALS="${GEPA_MAX_FULL_EVALS:-10}"
GEPA_NUM_THREADS="${GEPA_NUM_THREADS:-4}"
GEPA_REFLECTION_MINIBATCH_SIZE="${GEPA_REFLECTION_MINIBATCH_SIZE:-16}"
DEBUG="${GEPA_DEBUG:-${DEBUG:-false}}"
DEBUG_LIMIT="${GEPA_DEBUG_LIMIT:-${DEBUG_LIMIT:-4}}"
GEPA_SEEDS=(${GEPA_SEEDS_OVERRIDE:-18})

IFS=, read -r -a OPTIMIZE_TARGETS <<< "${OPTIMIZE_TARGETS_CSV}"
mkdir -p "${SCRIPT_DIR}/logs"

for gepa_seed in "${GEPA_SEEDS[@]}"; do
  gepa_run_dir="${GEPA_RUN_DIR:-${BASE_ENV_OUTPUT_DIR}_qwen_seed_${QWEN_SEED}_deepseek_seed_${DEEPSEEK_SEED}}_gepa_seed${gepa_seed}"
  safe_targets="${OPTIMIZE_TARGETS_CSV//[,._]/_}"
  cmd=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
    --algorithm gepa
    --input-jsonl "${INPUT_JSONL}"
    --trainset-jsonl "${TRAINSET_JSONL}"
    --valset-jsonl "${VALSET_JSONL}"
    --daily-omni-root "${DAILY_OMNI_ROOT}"
    --output-program "${gepa_run_dir}/compiled_gepa.json"
    --initial-program "${gepa_run_dir}/initial_gepa_program.json"
    --metadata-json "${gepa_run_dir}/compiled_gepa_metadata.json"
    --signature-search-json "${gepa_run_dir}/compiled_gepa_signature_search.json"
    --trajectory-jsonl "${gepa_run_dir}/optimized_trainset_trajectories.jsonl"
    --optimizer-log-dir "${gepa_run_dir}/optimizer_logs"
    --optimized-prompt-config-yaml "${gepa_run_dir}/${safe_targets}_GEPA_${GEPA_OPTIMIZED_PROMPT_CONFIG_BASENAME}"
    --optimization-train-only
    --max-turns "${MAX_TURNS}"
    --perception-model "${PERCEPTION_MODEL}"
    --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
    --captioner-config-yaml "${CAPTIONER_CONFIG_YAML}"
    --prompt-yaml "${PROMPT_YAML}"
    --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
    --signature-in-system-prompt
    --caption-placement task
    --train-limit "${TRAIN_LIMIT}"
    --gepa-max-full-evals "${GEPA_MAX_FULL_EVALS}"
    --gepa-num-threads "${GEPA_NUM_THREADS}"
    --gepa-reflection-minibatch-size "${GEPA_REFLECTION_MINIBATCH_SIZE}"
    --gepa-candidate-selection-strategy pareto
    --gepa-max-merge-invocations 5
    --gepa-seed "${gepa_seed}"
    --gepa-reflection-template-version "${GEPA_REFLECTION_TEMPLATE_VERSION}"
    --gepa-log-dir "${gepa_run_dir}/gepa_logs"
    --gepa-track-stats
    --gepa-track-best-outputs
    --caption-cache-dir "${CAPTION_CACHE_DIR}"
    --ignore-audio-caption-dir
  )
  for optimize_target in "${OPTIMIZE_TARGETS[@]}"; do
    cmd+=(--optimize-target "${optimize_target}")
  done
  if [ "${DEBUG}" = "true" ]; then
    cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
  fi

  echo "${SCRIPT_DIR}/logs/${GEPA_EXPERIMENT_NAME}.log"
  {
    echo "GEPA experiment=${GEPA_EXPERIMENT_NAME}; seed=${gepa_seed}; targets=${OPTIMIZE_TARGETS_CSV}"
    echo "Prompt=${PROMPT_YAML}; planner=${PLANNER_PROVIDER}/${DEEPSEEK_MODEL}; reflection=deepseek/${GEPA_REFLECTION_MODEL}"
    echo "DeepSeek planner base url=${DEEPSEEK_BASE_URL:-https://api.deepseek.com}; planner thinking=${PLANNER_THINKING_MODE}"
    echo "DeepSeek reflection base url=${GEPA_REFLECTION_API_BASE}; reflection thinking=${GEPA_REFLECTION_THINKING_MODE}"
    echo "Qwen perception config yaml=${PERCEPTION_CONFIG_YAML}"
    echo "Qwen perception base url=${QWEN_BASE_URL}"
    echo "Gemini caption cache dir=${CAPTION_CACHE_DIR}"
    echo "Perception=${PERCEPTION_MODEL}; captioner=${CAPTIONER_MODEL}; tools=${DSPY_AVQA_ALLOWED_TOOLS}; gepa_threads=${GEPA_NUM_THREADS}; qwen_batch_retries=${QWEN_RELIABLE_MAX_BATCH_RETRIES:-3}"
    "${cmd[@]}" "$@"
    PYTHONPATH="${PROJECT_DIR}/DSPy${PYTHONPATH:+:${PYTHONPATH}}" \
      "${PYTHON_BIN}" scripts/analyze_gepa_detailed.py --gepa-result-dir "${gepa_run_dir}"
  } >> "${SCRIPT_DIR}/logs/${GEPA_EXPERIMENT_NAME}.log" 2>&1
done
