#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_2.5_v0_corpo_cold}"

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

BASE_OUTPUT_ROOT="${MIPROV2_BASE_OUTPUT_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
INPUT_JSONL="${MIPROV2_INPUT_JSONL:-${INPUT_JSONL}}"
AUDIO_CAPTION_DIR="${MIPROV2_AUDIO_CAPTION_DIR:-${AUDIO_CAPTION_DIR}}"
MIPROV2_RUN_DIR="${MIPROV2_RUN_DIR:-${BASE_OUTPUT_ROOT}/daily_omni_dspy_gemini2.5_miprov2_recommended}"
INITIAL_PROGRAM="${MIPROV2_INITIAL_PROGRAM:-${MIPROV2_RUN_DIR}/initial_miprov2_program.json}"
OUTPUT_PROGRAM="${MIPROV2_OUTPUT_PROGRAM:-${MIPROV2_RUN_DIR}/compiled_miprov2.json}"
METADATA_JSON="${MIPROV2_METADATA_JSON:-${MIPROV2_RUN_DIR}/compiled_miprov2_metadata.json}"
SIGNATURE_SEARCH_JSON="${MIPROV2_SIGNATURE_SEARCH_JSON:-${MIPROV2_RUN_DIR}/compiled_miprov2_signature_search.json}"
TRAJECTORY_JSONL="${MIPROV2_TRAJECTORY_JSONL:-${MIPROV2_RUN_DIR}/output_test.jsonl}"
MAX_TURNS="${MIPROV2_MAX_TURNS:-${MAX_TURNS:-4}}"
PERCEPTION_MODEL="${MIPROV2_PERCEPTION_MODEL:-${PERCEPTION_MODEL:-gemini}}"
TRAIN_LIMIT="${MIPROV2_TRAIN_LIMIT:-64}"
MIPROV2_AUTO="${MIPROV2_AUTO:-none}"
MIPROV2_NUM_CANDIDATES="${MIPROV2_NUM_CANDIDATES:-6}"
MIPROV2_NUM_TRIALS="${MIPROV2_NUM_TRIALS:-10}"
MIPROV2_MAX_BOOTSTRAPPED_DEMOS="${MIPROV2_MAX_BOOTSTRAPPED_DEMOS:-2}"
MIPROV2_MAX_LABELED_DEMOS="${MIPROV2_MAX_LABELED_DEMOS:-2}"
MIPROV2_SEED="${MIPROV2_SEED:-9}"
MIPROV2_INIT_TEMPERATURE="${MIPROV2_INIT_TEMPERATURE:-1.0}"
MIPROV2_NUM_THREADS="${MIPROV2_NUM_THREADS:-}"
MIPROV2_MAX_ERRORS="${MIPROV2_MAX_ERRORS:-}"
MIPROV2_MINIBATCH="${MIPROV2_MINIBATCH:-true}"
MIPROV2_MINIBATCH_SIZE="${MIPROV2_MINIBATCH_SIZE:-16}"
MIPROV2_MINIBATCH_FULL_EVAL_STEPS="${MIPROV2_MINIBATCH_FULL_EVAL_STEPS:-5}"
MIPROV2_VIEW_DATA_BATCH_SIZE="${MIPROV2_VIEW_DATA_BATCH_SIZE:-8}"
DEBUG="${MIPROV2_DEBUG:-${DEBUG:-false}}"
DEBUG_LIMIT="${MIPROV2_DEBUG_LIMIT:-${DEBUG_LIMIT:-4}}"

# Print related settings to verify env loading without leaking secrets.
echo "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:+***set***}"
echo "DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-}"
echo "DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-}"
echo "QWEN_API_KEY=${QWEN_API_KEY:+***set***}"
echo "QWEN_BASE_URL=${QWEN_BASE_URL:-}"
echo "QWEN_MODEL=${QWEN_MODEL:-}"
echo "GEMINI_API_KEY=${GEMINI_API_KEY:+***set***}"
echo "GEMINI_BASE_URL=${GEMINI_BASE_URL:-}"
echo "GEMINI_MODEL=${GEMINI_MODEL:-}"
echo "PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML:-}"
echo "PROMPT_YAML=${PROMPT_YAML:-}"
echo "DSPY_AVQA_ALLOWED_TOOLS=${DSPY_AVQA_ALLOWED_TOOLS:-}"
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "AUDIO_CAPTION_DIR=${AUDIO_CAPTION_DIR}"
echo "MIPROV2_RUN_DIR=${MIPROV2_RUN_DIR}"
echo "INITIAL_PROGRAM=${INITIAL_PROGRAM}"
echo "OUTPUT_PROGRAM=${OUTPUT_PROGRAM}"
echo "METADATA_JSON=${METADATA_JSON}"
echo "SIGNATURE_SEARCH_JSON=${SIGNATURE_SEARCH_JSON}"
echo "TRAJECTORY_JSONL=${TRAJECTORY_JSONL}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "TRAIN_LIMIT=${TRAIN_LIMIT}"
echo "MIPROV2_AUTO=${MIPROV2_AUTO}"
echo "MIPROV2_NUM_CANDIDATES=${MIPROV2_NUM_CANDIDATES}"
echo "MIPROV2_NUM_TRIALS=${MIPROV2_NUM_TRIALS}"
echo "MIPROV2_MAX_BOOTSTRAPPED_DEMOS=${MIPROV2_MAX_BOOTSTRAPPED_DEMOS}"
echo "MIPROV2_MAX_LABELED_DEMOS=${MIPROV2_MAX_LABELED_DEMOS}"
echo "MIPROV2_SEED=${MIPROV2_SEED}"
echo "MIPROV2_INIT_TEMPERATURE=${MIPROV2_INIT_TEMPERATURE}"
echo "MIPROV2_NUM_THREADS=${MIPROV2_NUM_THREADS:-<dspy-default>}"
echo "MIPROV2_MAX_ERRORS=${MIPROV2_MAX_ERRORS:-<dspy-default>}"
echo "MIPROV2_MINIBATCH=${MIPROV2_MINIBATCH}"
echo "MIPROV2_MINIBATCH_SIZE=${MIPROV2_MINIBATCH_SIZE}"
echo "MIPROV2_MINIBATCH_FULL_EVAL_STEPS=${MIPROV2_MINIBATCH_FULL_EVAL_STEPS}"
echo "MIPROV2_VIEW_DATA_BATCH_SIZE=${MIPROV2_VIEW_DATA_BATCH_SIZE}"
echo "DEBUG=${DEBUG}"
echo "DEBUG_LIMIT=${DEBUG_LIMIT}"

echo "Running DSPy MIPROv2 optimizer..."
cmd=(
  "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
  --algorithm miprov2
  --input-jsonl "${INPUT_JSONL}"
  --audio-caption-dir "${AUDIO_CAPTION_DIR}"
  --output-program "${OUTPUT_PROGRAM}"
  --initial-program "${INITIAL_PROGRAM}"
  --metadata-json "${METADATA_JSON}"
  --signature-search-json "${SIGNATURE_SEARCH_JSON}"
  --trajectory-jsonl "${TRAJECTORY_JSONL}"
  --max-turns "${MAX_TURNS}"
  --perception-model "${PERCEPTION_MODEL}"
  --miprov2-auto "${MIPROV2_AUTO}"
  --miprov2-num-candidates "${MIPROV2_NUM_CANDIDATES}"
  --miprov2-num-trials "${MIPROV2_NUM_TRIALS}"
  --miprov2-max-bootstrapped-demos "${MIPROV2_MAX_BOOTSTRAPPED_DEMOS}"
  --miprov2-max-labeled-demos "${MIPROV2_MAX_LABELED_DEMOS}"
  --miprov2-seed "${MIPROV2_SEED}"
  --miprov2-init-temperature "${MIPROV2_INIT_TEMPERATURE}"
  --miprov2-minibatch-size "${MIPROV2_MINIBATCH_SIZE}"
  --miprov2-minibatch-full-eval-steps "${MIPROV2_MINIBATCH_FULL_EVAL_STEPS}"
  --miprov2-view-data-batch-size "${MIPROV2_VIEW_DATA_BATCH_SIZE}"
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

if [ -n "${MIPROV2_NUM_THREADS}" ]; then
  cmd+=(--miprov2-num-threads "${MIPROV2_NUM_THREADS}")
fi

if [ -n "${MIPROV2_MAX_ERRORS}" ]; then
  cmd+=(--miprov2-max-errors "${MIPROV2_MAX_ERRORS}")
fi

if [ "${MIPROV2_MINIBATCH}" != "true" ]; then
  cmd+=(--miprov2-no-minibatch)
fi

if [ "${DEBUG}" = "true" ]; then
  cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi

if [ "${MIPROV2_SKIP_BAD_EXAMPLES:-false}" = "true" ]; then
  cmd+=(--skip-bad-examples)
fi

cmd+=("$@")
"${cmd[@]}"

