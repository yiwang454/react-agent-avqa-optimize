#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_qwen_v2}"

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

# Keep the same starting runtime choices as run_dspy_react_agent_daily_qwen_v6_qwen_seed.sh.
export QWEN_SEED="${QWEN_SEED:-1234}"
export DEEPSEEK_SEED="${DEEPSEEK_SEED:-7}"
export PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa/DSPy/dspy_avqa/yamls/daily_qa_prompt_v6.yaml}"
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/general_scripts/react-agent-avqa/DSPy/dspy_avqa/yamls/config_localqwen_api_instruct.yaml}"
export QWEN_BASE_URL="${QWEN_BASE_URL_OVERRIDE:-http://10.62.2.189:8000/v1}"
export MAX_TURNS="${MIPROV2_MAX_TURNS:-${MAX_TURNS:-4}}"
export PERCEPTION_MODEL="${MIPROV2_PERCEPTION_MODEL:-qwen}"

BASE_ENV_OUTPUT_DIR="${MIPROV2_BASE_ENV_OUTPUT_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_qwen_v6MULTITURN_miprov2}"
INPUT_JSONL="${MIPROV2_INPUT_JSONL:-${INPUT_JSONL}}"
AUDIO_CAPTION_DIR="${MIPROV2_AUDIO_CAPTION_DIR:-${AUDIO_CAPTION_DIR}}"
MIPROV2_RUN_DIR="${MIPROV2_RUN_DIR:-${BASE_ENV_OUTPUT_DIR}_qwen_seed_${QWEN_SEED}_deepseek_seed_${DEEPSEEK_SEED}}"

INITIAL_PROGRAM="${MIPROV2_INITIAL_PROGRAM:-${MIPROV2_RUN_DIR}/initial_miprov2_program.json}"
OUTPUT_PROGRAM="${MIPROV2_OUTPUT_PROGRAM:-${MIPROV2_RUN_DIR}/compiled_miprov2.json}"
METADATA_JSON="${MIPROV2_METADATA_JSON:-${MIPROV2_RUN_DIR}/compiled_miprov2_metadata.json}"
SIGNATURE_SEARCH_JSON="${MIPROV2_SIGNATURE_SEARCH_JSON:-${MIPROV2_RUN_DIR}/compiled_miprov2_signature_search.json}"
TRAJECTORY_JSONL="${MIPROV2_TRAJECTORY_JSONL:-${MIPROV2_RUN_DIR}/output_test.jsonl}"
TRAIN_LIMIT="${MIPROV2_TRAIN_LIMIT:-64}"
DEBUG="${MIPROV2_DEBUG:-${DEBUG:-false}}"
DEBUG_LIMIT="${MIPROV2_DEBUG_LIMIT:-${DEBUG_LIMIT:-4}}"

mkdir -p "${MIPROV2_RUN_DIR}"
miprov2_config_file="${MIPROV2_RUN_DIR}/miprov2_config.json"
cat > "${miprov2_config_file}" <<'JSON'
{
  "auto": null,
  "num_candidates": 6,
  "num_trials": 10,
  "max_bootstrapped_demos": 2,
  "max_labeled_demos": 2,
  "seed": 9,
  "init_temperature": 1.0,
  "num_threads": null,
  "max_errors": null,
  "minibatch": true,
  "minibatch_size": 16,
  "minibatch_full_eval_steps": 5,
  "view_data_batch_size": 8
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
echo "MIPROV2_RUN_DIR=${MIPROV2_RUN_DIR}"
echo "MIPROV2_CONFIG=${miprov2_config_file}"
echo "INITIAL_PROGRAM=${INITIAL_PROGRAM}"
echo "OUTPUT_PROGRAM=${OUTPUT_PROGRAM}"
echo "METADATA_JSON=${METADATA_JSON}"
echo "SIGNATURE_SEARCH_JSON=${SIGNATURE_SEARCH_JSON}"
echo "TRAJECTORY_JSONL=${TRAJECTORY_JSONL}"
echo "MAX_TURNS=${MAX_TURNS}"
echo "TRAIN_LIMIT=${TRAIN_LIMIT}"
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
  --miprov2-config "${miprov2_config_file}"
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

if [ "${MIPROV2_SKIP_BAD_EXAMPLES:-false}" = "true" ]; then
  cmd+=(--skip-bad-examples)
fi

cmd+=("$@")
"${cmd[@]}"

