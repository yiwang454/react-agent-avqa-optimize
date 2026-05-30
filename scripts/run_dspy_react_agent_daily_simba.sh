#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_tool3_sanitycheck}"

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

INPUT_JSONL="${SIMBA_INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selected250.jsonl}"
AUDIO_CAPTION_DIR="${SIMBA_AUDIO_CAPTION_DIR:-${AUDIO_CAPTION_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_captioner_instruct}}"
SIMBA_RUN_DIR="${SIMBA_RUN_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_simba}"
OUTPUT_PROGRAM="${SIMBA_OUTPUT_PROGRAM:-${SIMBA_RUN_DIR}/compiled_simba.json}"
METADATA_JSON="${SIMBA_METADATA_JSON:-${SIMBA_RUN_DIR}/compiled_simba_metadata.json}"
MAX_TURNS="${SIMBA_MAX_TURNS:-${MAX_TURNS:-4}}"
PERCEPTION_MODEL="${SIMBA_PERCEPTION_MODEL:-${PERCEPTION_MODEL:-qwen}}"
TRAIN_LIMIT="${SIMBA_TRAIN_LIMIT:-250}"
SIMBA_BSIZE="${SIMBA_BSIZE:-32}"
SIMBA_NUM_CANDIDATES="${SIMBA_NUM_CANDIDATES:-6}"
SIMBA_MAX_STEPS="${SIMBA_MAX_STEPS:-8}"
SIMBA_MAX_DEMOS="${SIMBA_MAX_DEMOS:-4}"
SIMBA_NUM_THREADS="${SIMBA_NUM_THREADS:-}"
SIMBA_SEED="${SIMBA_SEED:-0}"
SIMBA_TEMPERATURE_FOR_SAMPLING="${SIMBA_TEMPERATURE_FOR_SAMPLING:-}"
SIMBA_TEMPERATURE_FOR_CANDIDATES="${SIMBA_TEMPERATURE_FOR_CANDIDATES:-}"
DEBUG="${SIMBA_DEBUG:-${DEBUG:-false}}"
DEBUG_LIMIT="${SIMBA_DEBUG_LIMIT:-${DEBUG_LIMIT:-4}}"

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
echo "OUTPUT_PROGRAM=${OUTPUT_PROGRAM}"
echo "METADATA_JSON=${METADATA_JSON}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "TRAIN_LIMIT=${TRAIN_LIMIT}"
echo "SIMBA_BSIZE=${SIMBA_BSIZE}"
echo "SIMBA_NUM_CANDIDATES=${SIMBA_NUM_CANDIDATES}"
echo "SIMBA_MAX_STEPS=${SIMBA_MAX_STEPS}"
echo "SIMBA_MAX_DEMOS=${SIMBA_MAX_DEMOS}"
echo "SIMBA_NUM_THREADS=${SIMBA_NUM_THREADS:-<dspy-default>}"
echo "SIMBA_SEED=${SIMBA_SEED}"
echo "SIMBA_TEMPERATURE_FOR_SAMPLING=${SIMBA_TEMPERATURE_FOR_SAMPLING:-<dspy-default>}"
echo "SIMBA_TEMPERATURE_FOR_CANDIDATES=${SIMBA_TEMPERATURE_FOR_CANDIDATES:-<dspy-default>}"
echo "DEBUG=${DEBUG}"
echo "DEBUG_LIMIT=${DEBUG_LIMIT}"

echo "Running DSPy SIMBA optimizer..."
cmd=(
  "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
  --algorithm simba
  --input-jsonl "${INPUT_JSONL}"
  --audio-caption-dir "${AUDIO_CAPTION_DIR}"
  --output-program "${OUTPUT_PROGRAM}"
  --metadata-json "${METADATA_JSON}"
  --max-turns "${MAX_TURNS}"
  --perception-model "${PERCEPTION_MODEL}"
  --simba-bsize "${SIMBA_BSIZE}"
  --simba-num-candidates "${SIMBA_NUM_CANDIDATES}"
  --simba-max-steps "${SIMBA_MAX_STEPS}"
  --simba-max-demos "${SIMBA_MAX_DEMOS}"
  --simba-seed "${SIMBA_SEED}"
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

if [ -n "${SIMBA_NUM_THREADS}" ]; then
  cmd+=(--simba-num-threads "${SIMBA_NUM_THREADS}")
fi

if [ -n "${SIMBA_TEMPERATURE_FOR_SAMPLING}" ]; then
  cmd+=(--simba-temperature-for-sampling "${SIMBA_TEMPERATURE_FOR_SAMPLING}")
fi

if [ -n "${SIMBA_TEMPERATURE_FOR_CANDIDATES}" ]; then
  cmd+=(--simba-temperature-for-candidates "${SIMBA_TEMPERATURE_FOR_CANDIDATES}")
fi

if [ "${DEBUG}" = "true" ]; then
  cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi

if [ "${SIMBA_SKIP_BAD_EXAMPLES:-false}" = "true" ]; then
  cmd+=(--skip-bad-examples)
fi

cmd+=("$@")
"${cmd[@]}"
