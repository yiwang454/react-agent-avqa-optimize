#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_qwen_v8}"
: "${GEPA_EXPERIMENT_NAME:?GEPA_EXPERIMENT_NAME must be set by the experiment wrapper}"
: "${OPTIMIZE_TARGETS_CSV:?OPTIMIZE_TARGETS_CSV must be set by the experiment wrapper}"

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

export PLANNER_PROVIDER="elm_gpt"
export PLANNER_MODEL="${PLANNER_MODEL_OVERRIDE:-gpt-5.4}"
export PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE_OVERRIDE:-0.0}"
export PLANNER_OUTPUT_SEQ_LEN="${PLANNER_OUTPUT_SEQ_LEN:-32768}"
export PLANNER_REASONING_EFFORT="${PLANNER_REASONING_EFFORT_OVERRIDE:-none}"
export PLANNER_SEED="${PLANNER_SEED_OVERRIDE:-1234}"
# export PLANNER_API_KEY="${PLANNER_API_KEY:-${ELM_API_KEY:-}}"
export PLANNER_API_KEY="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"
export DEEPSEEK_API_KEY="${PLANNER_API_KEY}"
unset DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
if [ -z "${PLANNER_API_KEY}" ]; then
  echo "Set PLANNER_API_KEY or ELM_API_KEY before running." >&2
  exit 2
fi

export PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task.yaml}"
export PERCEPTION_MODEL="${PERCEPTION_MODEL_OVERRIDE:-gemini}"
export CAPTIONER_MODEL="${CAPTIONER_MODEL_OVERRIDE:-gemini}"
GEMINI_RUNTIME_CONFIG_YAML="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${GEMINI_RUNTIME_CONFIG_YAML}}"
export CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML_OVERRIDE:-${GEMINI_RUNTIME_CONFIG_YAML}}"
export DSPY_AVQA_ALLOWED_TOOLS="${DSPY_AVQA_ALLOWED_TOOLS_OVERRIDE:-ask_caption,ask_perception}"
export GEMINI_API_BACKEND="legacy"
export MAX_TURNS="${GEPA_MAX_TURNS:-${MAX_TURNS:-3}}"

BASE_ENV_OUTPUT_DIR="${GEPA_BASE_ENV_OUTPUT_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_${GEPA_EXPERIMENT_NAME}}"
INPUT_JSONL="${GEPA_INPUT_JSONL:-${INPUT_JSONL}}"
DAILY_OMNI_ROOT="${GEPA_DAILY_OMNI_ROOT:-${DAILY_OMNI_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}}"
TRAINSET_JSONL="${GEPA_TRAINSET_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedTrain125.jsonl}"
VALSET_JSONL="${GEPA_VALSET_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedVal125.jsonl}"
AUDIO_CAPTION_DIR="${GEPA_AUDIO_CAPTION_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_captioner_seed27_gemini-2.5-flash_QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3_0.0/}"
TRAIN_LIMIT="${GEPA_TRAIN_LIMIT:-64}"
GEPA_MAX_FULL_EVALS="${GEPA_MAX_FULL_EVALS:-8}"
GEPA_REFLECTION_MINIBATCH_SIZE="${GEPA_REFLECTION_MINIBATCH_SIZE:-8}"
DEBUG="${GEPA_DEBUG:-${DEBUG:-false}}"
DEBUG_LIMIT="${GEPA_DEBUG_LIMIT:-${DEBUG_LIMIT:-4}}"
IFS=, read -r -a OPTIMIZE_TARGETS <<< "${OPTIMIZE_TARGETS_CSV}"
GEPA_SEEDS=(${GEPA_SEEDS_OVERRIDE:-18})

for gepa_seed in "${GEPA_SEEDS[@]}"; do
  gepa_run_dir="${GEPA_RUN_DIR:-${BASE_ENV_OUTPUT_DIR}_planner_${PLANNER_MODEL}_seed_${PLANNER_SEED}}_gepa_seed${gepa_seed}"
  safe_targets="${OPTIMIZE_TARGETS_CSV//[,._]/_}"
  cmd=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
    --algorithm gepa
    --input-jsonl "${INPUT_JSONL}"
    --trainset-jsonl "${TRAINSET_JSONL}"
    --valset-jsonl "${VALSET_JSONL}"
    --daily-omni-root "${DAILY_OMNI_ROOT}"
    --audio-caption-dir "${AUDIO_CAPTION_DIR}"
    --output-program "${gepa_run_dir}/compiled_gepa.json"
    --initial-program "${gepa_run_dir}/initial_gepa_program.json"
    --metadata-json "${gepa_run_dir}/compiled_gepa_metadata.json"
    --signature-search-json "${gepa_run_dir}/compiled_gepa_signature_search.json"
    --trajectory-jsonl "${gepa_run_dir}/optimized_trainset_trajectories.jsonl"
    --optimizer-log-dir "${gepa_run_dir}/optimizer_logs"
    --optimized-prompt-config-yaml "${gepa_run_dir}/${safe_targets}_GEPA_daily_qa_prompt_v8_caption_in_task.yaml"
    --final-eval-output-jsonl "${gepa_run_dir}/output_test.jsonl"
    --final-eval-output-dir "${gepa_run_dir}"
    --max-turns "${MAX_TURNS}"
    --perception-model "${PERCEPTION_MODEL}"
    --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
    --captioner-config-yaml "${CAPTIONER_CONFIG_YAML}"
    --prompt-yaml "${PROMPT_YAML}"
    --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
    --gemini-api-backend legacy
    --signature-in-system-prompt
    --caption-placement task
    --train-limit "${TRAIN_LIMIT}"
    --gepa-max-full-evals "${GEPA_MAX_FULL_EVALS}"
    --gepa-reflection-minibatch-size "${GEPA_REFLECTION_MINIBATCH_SIZE}"
    --gepa-candidate-selection-strategy pareto
    --gepa-max-merge-invocations 5
    --gepa-seed "${gepa_seed}"
    --gepa-log-dir "${gepa_run_dir}/gepa_logs"
    --gepa-track-stats
    --gepa-track-best-outputs
  )
  for optimize_target in "${OPTIMIZE_TARGETS[@]}"; do
    cmd+=(--optimize-target "${optimize_target}")
  done
  if [ "${DEBUG}" = "true" ]; then
    cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
  fi
  echo "GEPA experiment=${GEPA_EXPERIMENT_NAME}; seed=${gepa_seed}; targets=${OPTIMIZE_TARGETS_CSV}"
  echo "GEPA trainset=${TRAINSET_JSONL}; valset=${VALSET_JSONL}; final_eval_input=${INPUT_JSONL}"
  echo "GEPA max_full_evals=${GEPA_MAX_FULL_EVALS}; reflection_minibatch_size=${GEPA_REFLECTION_MINIBATCH_SIZE}"
  "${cmd[@]}" "$@"
done
