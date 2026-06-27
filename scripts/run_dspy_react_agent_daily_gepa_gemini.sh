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
INITIAL_PROGRAM="${GEPA_INITIAL_PROGRAM:-${GEPA_RUN_DIR}/initial_gepa_program.json}"
OUTPUT_PROGRAM="${GEPA_OUTPUT_PROGRAM:-${GEPA_RUN_DIR}/compiled_gepa.json}"
METADATA_JSON="${GEPA_METADATA_JSON:-${GEPA_RUN_DIR}/compiled_gepa_metadata.json}"
SIGNATURE_SEARCH_JSON="${GEPA_SIGNATURE_SEARCH_JSON:-${GEPA_RUN_DIR}/compiled_gepa_signature_search.json}"
TRAJECTORY_JSONL="${GEPA_TRAJECTORY_JSONL:-${GEPA_RUN_DIR}/output_test.jsonl}"
GEPA_LOG_DIR="${GEPA_LOG_DIR:-${GEPA_RUN_DIR}/gepa_logs}"
MAX_TURNS="${GEPA_MAX_TURNS:-${MAX_TURNS:-4}}"
PERCEPTION_MODEL="${GEPA_PERCEPTION_MODEL:-${PERCEPTION_MODEL:-gemini}}"
TRAIN_LIMIT="${GEPA_TRAIN_LIMIT:-64}"
GEPA_AUTO="${GEPA_AUTO:-none}"
GEPA_MAX_FULL_EVALS="${GEPA_MAX_FULL_EVALS:-6}"
GEPA_MAX_METRIC_CALLS="${GEPA_MAX_METRIC_CALLS:-}"
GEPA_REFLECTION_MINIBATCH_SIZE="${GEPA_REFLECTION_MINIBATCH_SIZE:-3}"
GEPA_CANDIDATE_SELECTION_STRATEGY="${GEPA_CANDIDATE_SELECTION_STRATEGY:-pareto}"
GEPA_SKIP_PERFECT_SCORE="${GEPA_SKIP_PERFECT_SCORE:-true}"
GEPA_USE_MERGE="${GEPA_USE_MERGE:-true}"
GEPA_MAX_MERGE_INVOCATIONS="${GEPA_MAX_MERGE_INVOCATIONS:-5}"
GEPA_NUM_THREADS="${GEPA_NUM_THREADS:-}"
GEPA_SEED="${GEPA_SEED:-0}"
GEPA_TRACK_STATS="${GEPA_TRACK_STATS:-false}"
DEBUG="${GEPA_DEBUG:-${DEBUG:-false}}"
DEBUG_LIMIT="${GEPA_DEBUG_LIMIT:-${DEBUG_LIMIT:-4}}"

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
echo "GEPA_RUN_DIR=${GEPA_RUN_DIR}"
echo "INITIAL_PROGRAM=${INITIAL_PROGRAM}"
echo "OUTPUT_PROGRAM=${OUTPUT_PROGRAM}"
echo "METADATA_JSON=${METADATA_JSON}"
echo "SIGNATURE_SEARCH_JSON=${SIGNATURE_SEARCH_JSON}"
echo "TRAJECTORY_JSONL=${TRAJECTORY_JSONL}"
echo "GEPA_LOG_DIR=${GEPA_LOG_DIR}"
echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "TRAIN_LIMIT=${TRAIN_LIMIT}"
echo "GEPA_AUTO=${GEPA_AUTO}"
echo "GEPA_MAX_FULL_EVALS=${GEPA_MAX_FULL_EVALS}"
echo "GEPA_MAX_METRIC_CALLS=${GEPA_MAX_METRIC_CALLS:-<unset>}"
echo "GEPA_REFLECTION_MINIBATCH_SIZE=${GEPA_REFLECTION_MINIBATCH_SIZE}"
echo "GEPA_CANDIDATE_SELECTION_STRATEGY=${GEPA_CANDIDATE_SELECTION_STRATEGY}"
echo "GEPA_SKIP_PERFECT_SCORE=${GEPA_SKIP_PERFECT_SCORE}"
echo "GEPA_USE_MERGE=${GEPA_USE_MERGE}"
echo "GEPA_MAX_MERGE_INVOCATIONS=${GEPA_MAX_MERGE_INVOCATIONS}"
echo "GEPA_NUM_THREADS=${GEPA_NUM_THREADS:-<dspy-default>}"
echo "GEPA_SEED=${GEPA_SEED}"
echo "GEPA_TRACK_STATS=${GEPA_TRACK_STATS}"
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
  --gepa-auto "${GEPA_AUTO}"
  --gepa-reflection-minibatch-size "${GEPA_REFLECTION_MINIBATCH_SIZE}"
  --gepa-candidate-selection-strategy "${GEPA_CANDIDATE_SELECTION_STRATEGY}"
  --gepa-max-merge-invocations "${GEPA_MAX_MERGE_INVOCATIONS}"
  --gepa-seed "${GEPA_SEED}"
  --gepa-log-dir "${GEPA_LOG_DIR}"
)

if [ -n "${TRAIN_LIMIT}" ]; then
  cmd+=(--train-limit "${TRAIN_LIMIT}")
fi

if [ -n "${GEPA_MAX_METRIC_CALLS}" ]; then
  cmd+=(--gepa-max-metric-calls "${GEPA_MAX_METRIC_CALLS}")
else
  cmd+=(--gepa-max-full-evals "${GEPA_MAX_FULL_EVALS}")
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

if [ "${GEPA_SKIP_PERFECT_SCORE}" != "true" ]; then
  cmd+=(--gepa-dont-skip-perfect-score)
fi

if [ "${GEPA_USE_MERGE}" != "true" ]; then
  cmd+=(--gepa-no-merge)
fi

if [ -n "${GEPA_NUM_THREADS}" ]; then
  cmd+=(--gepa-num-threads "${GEPA_NUM_THREADS}")
fi

if [ "${GEPA_TRACK_STATS}" = "true" ]; then
  cmd+=(--gepa-track-stats)
fi

if [ "${DEBUG}" = "true" ]; then
  cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
fi

if [ "${GEPA_SKIP_BAD_EXAMPLES:-false}" = "true" ]; then
  cmd+=(--skip-bad-examples)
fi

cmd+=("$@")
"${cmd[@]}"

