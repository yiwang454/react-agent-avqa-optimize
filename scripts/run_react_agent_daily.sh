#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_langchain_react_agent_daily}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_PREFIX}"
cd "${PROJECT_DIR}"

# Ensure uv is installed before usage.
if ! command -v uv >/dev/null 2>&1; then
  python -m pip install -U uv
fi

# Ensure API/runtime helpers are installed before usage.
if ! python -c "import requests" >/dev/null 2>&1; then
  python -m pip install -U requests
fi
if ! python -c "import openai" >/dev/null 2>&1; then
  python -m pip install -U openai
fi
if ! python -c "import yaml" >/dev/null 2>&1; then
  python -m pip install -U pyyaml
fi

# Print related settings to verify env loading without leaking secrets.
echo "DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY:+***set***}"
echo "DEEPSEEK_TOKEN=${DEEPSEEK_TOKEN:+***set***}"
echo "DEEPSEEK_BASE_URL=${DEEPSEEK_BASE_URL:-}"
echo "DEEPSEEK_MODEL=${DEEPSEEK_MODEL}"
echo "PLANNER_API_KEY=${PLANNER_API_KEY:+***set***}"
echo "PLANNER_API_BASE=${PLANNER_API_BASE:-}"
echo "PLANNER_MODEL=${PLANNER_MODEL:-}"
echo "PLANNER_RETRY_DELAY_S=${PLANNER_RETRY_DELAY_S:-}"
echo "PLANNER_OUTPUT_SEQ_LEN=${PLANNER_OUTPUT_SEQ_LEN:-}"
echo "PLANNER_MAX_INPUT_SEQ_LEN=${PLANNER_MAX_INPUT_SEQ_LEN:-}"
echo "PLANNER_INTENT_PLUGIN_ID=${PLANNER_INTENT_PLUGIN_ID:-}"
echo "PLANNER_DECOUPLED=${PLANNER_DECOUPLED:-}"
echo "QWEN_API_KEY=${QWEN_API_KEY:+***set***}"
echo "GEMINI_API_KEY=${GEMINI_API_KEY:+***set***}"
echo "GEMINI_BASE_URL=${GEMINI_BASE_URL}"
echo "GEMINI_MODEL=${GEMINI_MODEL}"
echo "GEMINI_VIDEO_ONLY=${GEMINI_VIDEO_ONLY:-}"
echo "GEMINI_TIMEOUT=${GEMINI_TIMEOUT:-}"
echo "GEMINI_MAX_RETRIES=${GEMINI_MAX_RETRIES:-}"
echo "GEMINI_RETRY_DELAY_S=${GEMINI_RETRY_DELAY_S:-}"
echo "GEMINI_TEMPERATURE=${GEMINI_TEMPERATURE:-}"
echo "GEMINI_TOP_P=${GEMINI_TOP_P:-}"
echo "GEMINI_TOP_K=${GEMINI_TOP_K:-}"
echo "GEMINI_MAX_TOKENS=${GEMINI_MAX_TOKENS:-}"
echo "GEMINI_INCLUDE_THOUGHTS=${GEMINI_INCLUDE_THOUGHTS:-}"
echo "GEMINI_RETURN_THINKING=${GEMINI_RETURN_THINKING:-}"
echo "PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML:-}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
echo "DEBUG=${DEBUG}"

echo "Running batch runner..."
cmd=(
  uv run --no-sync python -m react_agent.batch_runner
  --input-jsonl "${INPUT_JSONL}"
  --audio-caption-dir "${AUDIO_CAPTION_DIR}"
  --output-jsonl "${OUTPUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --perception-model "${PERCEPTION_MODEL}"
  --max-turns "${MAX_TURNS}"
  --recursion-limit "${RECURSION_LIMIT}"
  --concurrency "${CONCURRENCY}"
)

if [ -n "${PERCEPTION_CONFIG_YAML:-}" ]; then
  cmd+=(--perception-config-yaml "${PERCEPTION_CONFIG_YAML}")
fi

if [ "${DEBUG}" = "true" ]; then
  cmd+=(--debug)
fi

cmd+=("$@")
"${cmd[@]}"
