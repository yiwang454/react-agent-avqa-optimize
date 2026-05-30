#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_2.5_v0_corpo_cold"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

cd "${PROJECT_DIR}"


BASE_OUTPUT_ROOT="/mnt/ceph_rbd/data/avqa_project/daily_omni"

run_copro() {
  local preset="$1"
  local output_dir="$2"
  local train_limit="$3"
  local breadth="$4"
  local depth="$5"
  local init_temperature="$6"

  local initial_program="${output_dir}/initial_copro_program.json"
  local output_program="${output_dir}/compiled_copro.json"
  local metadata_json="${output_dir}/compiled_copro_metadata.json"
  local signature_search_json="${output_dir}/compiled_copro_signature_search.json"
  local trajectory_jsonl="${output_dir}/output_test.jsonl"

  echo "============================================================"
  echo "Running DSPy COPRO optimizer: ${preset}"
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
  echo "OUTPUT_DIR=${output_dir}"
  echo "INITIAL_PROGRAM=${initial_program}"
  echo "OUTPUT_PROGRAM=${output_program}"
  echo "METADATA_JSON=${metadata_json}"
  echo "SIGNATURE_SEARCH_JSON=${signature_search_json}"
  echo "OUTPUT_JSONL=${trajectory_jsonl}"
  echo "PERCEPTION_MODEL=${PERCEPTION_MODEL}"
  echo "MAX_TURNS=${MAX_TURNS}"
  echo "COPRO_PRESET=${preset}"
  echo "TRAIN_LIMIT=${train_limit}"
  echo "COPRO_BREADTH=${breadth}"
  echo "COPRO_DEPTH=${depth}"
  echo "COPRO_INIT_TEMPERATURE=${init_temperature}"
  echo "PLANNER_OUTPUT_SEQ_LEN=${PLANNER_OUTPUT_SEQ_LEN}"
  
  cmd=(
    python DSPy/avqa_dspy_optimize.py
    --algorithm copro
    --input-jsonl "${INPUT_JSONL}"
    --audio-caption-dir "${AUDIO_CAPTION_DIR}"
    --output-program "${output_program}"
    --initial-program "${initial_program}"
    --metadata-json "${metadata_json}"
    --signature-search-json "${signature_search_json}"
    --trajectory-jsonl "${trajectory_jsonl}"
    --max-turns "${MAX_TURNS}"
    --perception-model "${PERCEPTION_MODEL}"
    --train-limit "${train_limit}"
    --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
    --prompt-yaml "${PROMPT_YAML}"
    --copro-breadth "${breadth}"
    --copro-depth "${depth}"
    --copro-init-temperature "${init_temperature}"
  )

  if [ "${DEBUG}" = "true" ]; then
    cmd+=(--debug --debug-limit "${DEBUG_LIMIT}")
  fi

  "${cmd[@]}"
}

# run_copro \
#   pilot \
#   "${BASE_OUTPUT_ROOT}/daily_omni_dspy_gemini2.5_copro_cold_pilot" \
#   16 \
#   2 \
#   2 \
#   1.0

run_copro \
  recommended \
  "${BASE_OUTPUT_ROOT}/daily_omni_dspy_gemini2.5_copro_cold_recommended" \
  64 \
  5 \
  3 \
  1.0
