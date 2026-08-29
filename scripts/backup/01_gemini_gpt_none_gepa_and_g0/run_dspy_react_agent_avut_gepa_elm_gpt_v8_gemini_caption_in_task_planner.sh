#!/usr/bin/env bash
set -euo pipefail

# Run AVUT final-test inference for both artifacts from the v8 Gemini-caption-in-task
# planner GEPA experiment:
#   * best_candidate: compiled_gepa.json + GEPA-optimized prompt YAML
#   * v8_baseline:    initial_gepa_program.json + original v8 prompt YAML
#
# Both runs use DSPy's --inference-only mode.  Its per-sample cache makes a rerun
# resumable, so do not share either output directory with a different program.

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

# Match scripts/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_planner.sh.
export PLANNER_PROVIDER="elm_gpt"
export PLANNER_MODEL="${PLANNER_MODEL_OVERRIDE:-gpt-4.1}"
export PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE_OVERRIDE:-0.0}"
export PLANNER_OUTPUT_SEQ_LEN="${PLANNER_OUTPUT_SEQ_LEN:-32768}"
export PLANNER_REASONING_EFFORT="${PLANNER_REASONING_EFFORT_OVERRIDE:-none}"
export PLANNER_SEED="${PLANNER_SEED_OVERRIDE:-1234}"
# Do not place API credentials in an experiment script.  Supply either variable
# in the calling environment, for example: export ELM_API_KEY=... .
export PLANNER_API_KEY="${PLANNER_API_KEY:-${ELM_API_KEY:-}}"
export DEEPSEEK_API_KEY="${PLANNER_API_KEY}"
unset DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
if [ -z "${PLANNER_API_KEY}" ]; then
  echo "Set PLANNER_API_KEY or ELM_API_KEY before running." >&2
  exit 2
fi

export PERCEPTION_MODEL="${PERCEPTION_MODEL_OVERRIDE:-gemini}"
export CAPTIONER_MODEL="${CAPTIONER_MODEL_OVERRIDE:-gemini}"
GEMINI_RUNTIME_CONFIG_YAML="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${GEMINI_RUNTIME_CONFIG_YAML}}"
export CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML_OVERRIDE:-${GEMINI_RUNTIME_CONFIG_YAML}}"
export DSPY_AVQA_ALLOWED_TOOLS="${DSPY_AVQA_ALLOWED_TOOLS_OVERRIDE:-ask_caption,ask_perception}"
export GEMINI_API_BACKEND="legacy"
export MAX_TURNS="${GEPA_MAX_TURNS:-${MAX_TURNS:-3}}"

# AVUT input and the same caption source used by the GEPA experiment.
INPUT_JSONL="${AVUT_INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/avut_jsonls/avut_cuts_human_av_seperated.jsonl}"
AUDIO_CAPTION_DIR="${GEPA_AUDIO_CAPTION_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_captioner_seed27_gemini-2.5-flash_QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3_0.0/}"
# AVUT has no matching precomputed Daily captions.  The initial
# video_description is only a placeholder; ask_caption supplies the real caption.
IGNORE_AUDIO_CAPTION_DIR="${IGNORE_AUDIO_CAPTION_DIR:-true}"

# These defaults identify the completed GEPA run started by the planner wrapper.
GEPA_SEED="${GEPA_SEED:-18}"
GEPA_BASE_ENV_OUTPUT_DIR="${GEPA_BASE_ENV_OUTPUT_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_planner_workflow_prompt}"
GEPA_RUN_DIR="${GEPA_RUN_DIR_OVERRIDE:-${GEPA_BASE_ENV_OUTPUT_DIR}_planner_${PLANNER_MODEL}_seed_${PLANNER_SEED}_gepa_seed${GEPA_SEED}}"
GEPA_COMPILED_PROGRAM="${GEPA_COMPILED_PROGRAM_OVERRIDE:-${GEPA_RUN_DIR}/compiled_gepa.json}"
GEPA_INITIAL_PROGRAM="${GEPA_INITIAL_PROGRAM_OVERRIDE:-${GEPA_RUN_DIR}/initial_gepa_program.json}"
GEPA_OPTIMIZED_PROMPT_YAML="${GEPA_OPTIMIZED_PROMPT_YAML_OVERRIDE:-${GEPA_RUN_DIR}/planner_workflow_prompt_GEPA_daily_qa_prompt_v8_caption_in_task.yaml}"
BASELINE_PROMPT_YAML="${BASELINE_PROMPT_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task.yaml}"

# Keep the two compiled programs in distinct cache directories.  OUTPUT_BASE_DIR
# may be overridden when evaluating a different GEPA run.
OUTPUT_BASE_DIR="${AVUT_OUTPUT_BASE_DIR:-/mnt/ceph_rbd/data/avqa_project/avut/avut_dspy_GPT_v8GeminiCaptionInTask_gepa_planner_workflow_prompt_planner_${PLANNER_MODEL}_seed_${PLANNER_SEED}_gepa_seed${GEPA_SEED}}"
BEST_CANDIDATE_OUTPUT_DIR="${BEST_CANDIDATE_OUTPUT_DIR_OVERRIDE:-${OUTPUT_BASE_DIR}/best_candidate}"
BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR_OVERRIDE:-${OUTPUT_BASE_DIR}/v8_baseline}"

# Set RUN_MODE to best_candidate, baseline, or both (the default).
RUN_MODE="${RUN_MODE:-both}"
DEBUG="${GEPA_DEBUG:-${DEBUG:-false}}"
DEBUG_LIMIT="${GEPA_DEBUG_LIMIT:-${DEBUG_LIMIT:-4}}"
SCRIPT_ARGS=("$@")
child_pids=()

for required_path in \
  "${INPUT_JSONL}" \
  "${GEPA_COMPILED_PROGRAM}" \
  "${GEPA_INITIAL_PROGRAM}" \
  "${GEPA_OPTIMIZED_PROMPT_YAML}" \
  "${BASELINE_PROMPT_YAML}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

case "${RUN_MODE}" in
  best_candidate|baseline|both) ;;
  *)
    echo "RUN_MODE must be best_candidate, baseline, or both; got: ${RUN_MODE}" >&2
    exit 2
    ;;
esac

echo "PLANNER_PROVIDER=${PLANNER_PROVIDER}"
echo "PLANNER_MODEL=${PLANNER_MODEL}"
echo "PLANNER_API_KEY=${PLANNER_API_KEY:+***set***}"
echo "PLANNER_REASONING_EFFORT=${PLANNER_REASONING_EFFORT}"
echo "PLANNER_TEMPERATURE=${PLANNER_TEMPERATURE}"
echo "PLANNER_OUTPUT_SEQ_LEN=${PLANNER_OUTPUT_SEQ_LEN}"
echo "PLANNER_SEED=${PLANNER_SEED}"
echo "GEMINI_API_BACKEND=${GEMINI_API_BACKEND}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
echo "CAPTIONER_MODEL=${CAPTIONER_MODEL}"
echo "PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML}"
echo "CAPTIONER_CONFIG_YAML=${CAPTIONER_CONFIG_YAML}"
echo "DSPY_AVQA_ALLOWED_TOOLS=${DSPY_AVQA_ALLOWED_TOOLS}"
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "AUDIO_CAPTION_DIR=${AUDIO_CAPTION_DIR}"
echo "IGNORE_AUDIO_CAPTION_DIR=${IGNORE_AUDIO_CAPTION_DIR}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "DEBUG=${DEBUG}"
echo "DEBUG_LIMIT=${DEBUG_LIMIT}"
echo "RUN_MODE=${RUN_MODE}"

run_inference() {
  local run_name="$1"
  local program_path="$2"
  local prompt_yaml="$3"
  local output_dir="$4"
  local output_jsonl="${output_dir}/output_test.jsonl"

  echo "Running AVUT ${run_name} inference."
  echo "OUTPUT_PROGRAM=${program_path}"
  echo "PROMPT_YAML=${prompt_yaml}"
  echo "FINAL_EVAL_OUTPUT_DIR=${output_dir}"
  echo "FINAL_EVAL_OUTPUT_JSONL=${output_jsonl}"

  local -a cmd=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
    --algorithm gepa
    --inference-only
    --input-jsonl "${INPUT_JSONL}"
    --output-program "${program_path}"
    --final-eval-output-jsonl "${output_jsonl}"
    --final-eval-output-dir "${output_dir}"
    --max-turns "${MAX_TURNS}"
    --perception-model "${PERCEPTION_MODEL}"
    --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
    --captioner-config-yaml "${CAPTIONER_CONFIG_YAML}"
    --prompt-yaml "${prompt_yaml}"
    --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
    --gemini-api-backend legacy
    --signature-in-system-prompt
    --caption-placement task
  )

  if [ -n "${AUDIO_CAPTION_DIR:-}" ]; then
    cmd+=(--audio-caption-dir "${AUDIO_CAPTION_DIR}")
  fi

  if [ "${IGNORE_AUDIO_CAPTION_DIR}" = "true" ]; then
    cmd+=(--ignore-audio-caption-dir)
  fi

  if [ "${DEBUG}" = "true" ]; then
    cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
  fi

  cmd+=("${SCRIPT_ARGS[@]}")
  "${cmd[@]}"
}

cleanup_children() {
  local reason="${1:-exit}"
  trap - INT TERM EXIT

  if [ "${#child_pids[@]}" -gt 0 ]; then
    echo "Stopping ${#child_pids[@]} background AVUT inference process(es) after ${reason}..." >&2
    kill -TERM "${child_pids[@]}" 2>/dev/null || true
    wait "${child_pids[@]}" 2>/dev/null || true
  fi
}

trap 'cleanup_children SIGINT; exit 130' INT
trap 'cleanup_children SIGTERM; exit 143' TERM
trap 'cleanup_children EXIT' EXIT

if [ "${RUN_MODE}" = "best_candidate" ] || [ "${RUN_MODE}" = "both" ]; then
  echo "BEST_CANDIDATE_PROGRAM=${GEPA_COMPILED_PROGRAM}"
  echo "BEST_CANDIDATE_OUTPUT_DIR=${BEST_CANDIDATE_OUTPUT_DIR}"
  run_inference \
    "best_candidate" \
    "${GEPA_COMPILED_PROGRAM}" \
    "${GEPA_OPTIMIZED_PROMPT_YAML}" \
    "${BEST_CANDIDATE_OUTPUT_DIR}" &
  child_pids+=("$!")
fi

if [ "${RUN_MODE}" = "baseline" ] || [ "${RUN_MODE}" = "both" ]; then
  echo "BASELINE_PROGRAM=${GEPA_INITIAL_PROGRAM}"
  echo "BASELINE_OUTPUT_DIR=${BASELINE_OUTPUT_DIR}"
  run_inference \
    "v8_baseline" \
    "${GEPA_INITIAL_PROGRAM}" \
    "${BASELINE_PROMPT_YAML}" \
    "${BASELINE_OUTPUT_DIR}" &
  child_pids+=("$!")
fi

wait "${child_pids[@]}"
child_pids=()
trap - INT TERM EXIT
