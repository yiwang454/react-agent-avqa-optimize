#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_2.5_v0_corpo"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

cd "${PROJECT_DIR}"

INITIAL_PROGRAM="${OUTPUT_DIR}/initial_copro_program.json"
OUTPUT_PROGRAM="${OUTPUT_DIR}/compiled_copro.json"
METADATA_JSON="${OUTPUT_DIR}/compiled_copro_metadata.json"
SIGNATURE_SEARCH_JSON="${OUTPUT_DIR}/compiled_copro_signature_search.json"
TRAJECTORY_JSONL="${OUTPUT_DIR}/compiled_copro_train_trajectories.jsonl"

TRAIN_LIMIT="${COPRO_TRAIN_LIMIT:-16}"
COPRO_BREADTH="${COPRO_BREADTH:-10}"
COPRO_DEPTH="${COPRO_DEPTH:-3}"
COPRO_INIT_TEMPERATURE="${COPRO_INIT_TEMPERATURE:-1.0}"

# Print related settings to verify env loading without leaking secrets.
echo "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:+***set***}"
echo "DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-}"
echo "DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-}"
echo "GEMINI_API_KEY=${GEMINI_API_KEY:+***set***}"
echo "GEMINI_BASE_URL=${GEMINI_BASE_URL:-}"
echo "GEMINI_MODEL=${GEMINI_MODEL:-}"
echo "PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML}"
echo "PROMPT_YAML=${PROMPT_YAML}"
echo "INPUT_JSONL=${INPUT_JSONL}"
echo "AUDIO_CAPTION_DIR=${AUDIO_CAPTION_DIR}"
echo "INITIAL_PROGRAM=${INITIAL_PROGRAM}"
echo "OUTPUT_PROGRAM=${OUTPUT_PROGRAM}"
echo "METADATA_JSON=${METADATA_JSON}"
echo "SIGNATURE_SEARCH_JSON=${SIGNATURE_SEARCH_JSON}"
echo "TRAJECTORY_JSONL=${TRAJECTORY_JSONL}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "TRAIN_LIMIT=${TRAIN_LIMIT}"
echo "COPRO_BREADTH=${COPRO_BREADTH}"
echo "COPRO_DEPTH=${COPRO_DEPTH}"
echo "COPRO_INIT_TEMPERATURE=${COPRO_INIT_TEMPERATURE}"

echo "Running DSPy COPRO optimizer..."
cmd=(
  python DSPy/avqa_dspy_optimize.py
  --algorithm copro
  --input-jsonl "${INPUT_JSONL}"
  --audio-caption-dir "${AUDIO_CAPTION_DIR}"
  --output-program "${OUTPUT_PROGRAM}"
  --initial-program "${INITIAL_PROGRAM}"
  --metadata-json "${METADATA_JSON}"
  --signature-search-json "${SIGNATURE_SEARCH_JSON}"
  --trajectory-jsonl "${TRAJECTORY_JSONL}"
  --max-turns "${MAX_TURNS}"
  --perception-model "${PERCEPTION_MODEL}"
  --train-limit "${TRAIN_LIMIT}"
  --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
  --prompt-yaml "${PROMPT_YAML}"
  --copro-breadth "${COPRO_BREADTH}"
  --copro-depth "${COPRO_DEPTH}"
  --copro-init-temperature "${COPRO_INIT_TEMPERATURE}"
)

if [ "${DEBUG}" = "true" ]; then
  cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi

"${cmd[@]}"
