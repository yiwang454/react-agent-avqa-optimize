#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_key2_example_repetition}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

# conda activate "${ENV_PREFIX}"
cd "${PROJECT_DIR}"

# Print related settings to verify env loading without leaking secrets.
echo "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:+***set***}"
echo "DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL}"
echo "DEEPSEEK_MODEL=${DEEPSEEK_MODEL}"
echo "GEMINI_API_KEY=${GEMINI_API_KEY:+***set***}"
echo "GEMINI_BASE_URL=${GEMINI_BASE_URL}"
echo "GEMINI_MODEL=${GEMINI_MODEL}"
echo "PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML:-}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
echo "DEBUG=${DEBUG}"
echo "DEBUG_LIMIT=${DEBUG_LIMIT}"

echo "Running DSPy batch runner..."
cmd=(
  python DSPy/avqa_dspy_impl.py
  --input-jsonl "${INPUT_JSONL}"
  --audio-caption-dir "${AUDIO_CAPTION_DIR}"
  --output-jsonl "${OUTPUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --max-turns "${MAX_TURNS}"
  --perception-model "${PERCEPTION_MODEL}"
)

if [ -n "${PERCEPTION_CONFIG_YAML:-}" ]; then
  cmd+=(--perception-config-yaml "${PERCEPTION_CONFIG_YAML}")
fi

if [ "${DEBUG}" = "true" ]; then
  cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi

cmd+=("$@")
"${cmd[@]}"
