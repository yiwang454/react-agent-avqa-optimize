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

BASE_OUTPUT_ROOT="${GEPA_BASE_OUTPUT_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
INPUT_JSONL="${GEPA_INPUT_JSONL:-${INPUT_JSONL}}"
AUDIO_CAPTION_DIR="${GEPA_AUDIO_CAPTION_DIR:-${AUDIO_CAPTION_DIR}}"
GEPA_RUN_DIR="${GEPA_RUN_DIR:-${BASE_OUTPUT_ROOT}/daily_omni_dspy_gemini2.5_gepa_recommended}"
MAX_TURNS="${GEPA_MAX_TURNS:-${MAX_TURNS:-4}}"
PERCEPTION_MODEL="${GEPA_PERCEPTION_MODEL:-${PERCEPTION_MODEL:-gemini}}"

INITIAL_PROGRAM="${GEPA_INITIAL_PROGRAM:-${GEPA_RUN_DIR}/initial_gepa_program.json}"
OUTPUT_PROGRAM="${GEPA_OUTPUT_PROGRAM:-${GEPA_RUN_DIR}/compiled_gepa.json}"
METADATA_JSON="${GEPA_METADATA_JSON:-${GEPA_RUN_DIR}/compiled_gepa_metadata.json}"
SIGNATURE_SEARCH_JSON="${GEPA_SIGNATURE_SEARCH_JSON:-${GEPA_RUN_DIR}/compiled_gepa_signature_search.json}"
TRAJECTORY_JSONL="${GEPA_TRAJECTORY_JSONL:-${GEPA_RUN_DIR}/output_test.jsonl}"
TRAIN_LIMIT="${GEPA_TRAIN_LIMIT:-64}"
DEBUG="${GEPA_DEBUG:-${DEBUG:-false}}"
DEBUG_LIMIT="${GEPA_DEBUG_LIMIT:-${DEBUG_LIMIT:-4}}"

mkdir -p "${GEPA_RUN_DIR}"
gepa_config_file="${GEPA_RUN_DIR}/gepa_config.json"
gepa_log_dir="${GEPA_RUN_DIR}/gepa_logs"
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
  "seed": 0,
  "log_dir": "${gepa_log_dir}",
  "track_stats": false
}
JSON

# Print related settings to verify env loading without leaking secrets.
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
echo "GEPA_RUN_DIR=${GEPA_RUN_DIR}"
echo "GEPA_CONFIG=${gepa_config_file}"
echo "INITIAL_PROGRAM=${INITIAL_PROGRAM}"
echo "OUTPUT_PROGRAM=${OUTPUT_PROGRAM}"
echo "METADATA_JSON=${METADATA_JSON}"
echo "SIGNATURE_SEARCH_JSON=${SIGNATURE_SEARCH_JSON}"
echo "TRAJECTORY_JSONL=${TRAJECTORY_JSONL}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "TRAIN_LIMIT=${TRAIN_LIMIT}"
echo "DEBUG=${DEBUG}"
echo "DEBUG_LIMIT=${DEBUG_LIMIT}"

echo "Running DSPy GEPA optimizer..."
cmd=(
  "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
  --algorithm gepa
  --input-jsonl "${INPUT_JSONL}"
  --audio-caption-dir "${AUDIO_CAPTION_DIR}"
  --output-program "${OUTPUT_PROGRAM}"
  --initial-program "${INITIAL_PROGRAM}"
  --metadata-json "${METADATA_JSON}"
  --signature-search-json "${SIGNATURE_SEARCH_JSON}"
  --trajectory-jsonl "${TRAJECTORY_JSONL}"
  --max-turns "${MAX_TURNS}"
  --perception-model "${PERCEPTION_MODEL}"
  --gepa-config "${gepa_config_file}"
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

if [ "${GEPA_SKIP_BAD_EXAMPLES:-false}" = "true" ]; then
  cmd+=(--skip-bad-examples)
fi

cmd+=("$@")
"${cmd[@]}"

