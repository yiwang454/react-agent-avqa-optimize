#!/usr/bin/env bash
set -euo pipefail

# Re-run final inference for the best program from the completed G3 GPT-5.4
# planner GEPA experiment.  This script intentionally does not pass any GEPA
# optimization settings: --inference-only loads the already selected program.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_qwen_v8}"

# Original completed G3 GPT-5.4 planner run (GEPA seed 18).  Its saved
# compiled_gepa.json is candidate 6, selected on the full validation set.
GEPA_RUN_DIR="${GEPA_RUN_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_densified_planner_workflow_prompt_gpt5_4_medium_planner_gpt-5.4_seed_1234_gepa_seed18}"
COMPILED_PROGRAM="${GEPA_RUN_DIR}/compiled_gepa.json"
INFERENCE_OUTPUT_DIR="${GEPA_RUN_DIR}/dailyomni_inference_repeat"
LOG_FILE="${INFERENCE_OUTPUT_DIR}/run.log"

if [ ! -f "${COMPILED_PROGRAM}" ]; then
  echo "Missing completed GEPA program: ${COMPILED_PROGRAM}" >&2
  exit 2
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

# Preserve the original experiment's runtime configuration.  These values
# apply to the loaded candidate, not to GEPA optimization or reflection.
export PLANNER_PROVIDER="elm_gpt"
export PLANNER_MODEL="gpt-5.4"
export PLANNER_TEMPERATURE="0.0"
export PLANNER_OUTPUT_SEQ_LEN="32768"
export PLANNER_REASONING_EFFORT="none"
export PLANNER_SEED="1234"
export PLANNER_API_KEY="${PLANNER_API_KEY:-${ELM_API_KEY:-}}"
export DEEPSEEK_API_KEY="${PLANNER_API_KEY}"
unset DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
if [ -z "${PLANNER_API_KEY}" ]; then
  echo "Set PLANNER_API_KEY or ELM_API_KEY before running." >&2
  exit 2
fi

INPUT_JSONL="${INFERENCE_INPUT_JSONL:-${INPUT_JSONL:?INPUT_JSONL must be set by ${ENV_FILE}}}"
DAILY_OMNI_ROOT="${INFERENCE_DAILY_OMNI_ROOT:-${DAILY_OMNI_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}}"
AUDIO_CAPTION_DIR="${INFERENCE_AUDIO_CAPTION_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_captioner_seed27_gemini-2.5-flash_QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3_0.0/}"
PROMPT_YAML="${INFERENCE_PROMPT_YAML:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task.yaml}"
GEMINI_RUNTIME_CONFIG_YAML="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
PERCEPTION_CONFIG_YAML="${INFERENCE_PERCEPTION_CONFIG_YAML:-${GEMINI_RUNTIME_CONFIG_YAML}}"
CAPTIONER_CONFIG_YAML="${INFERENCE_CAPTIONER_CONFIG_YAML:-${GEMINI_RUNTIME_CONFIG_YAML}}"

export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="legacy"

mkdir -p "${INFERENCE_OUTPUT_DIR}"

cmd=(
  "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
  --algorithm gepa
  --inference-only
  --input-jsonl "${INPUT_JSONL}"
  --daily-omni-root "${DAILY_OMNI_ROOT}"
  --audio-caption-dir "${AUDIO_CAPTION_DIR}"
  --output-program "${COMPILED_PROGRAM}"
  --metadata-json "${INFERENCE_OUTPUT_DIR}/inference_metadata.json"
  --final-eval-output-jsonl "${INFERENCE_OUTPUT_DIR}/output_test.jsonl"
  --final-eval-output-dir "${INFERENCE_OUTPUT_DIR}"
  --final-eval-num-threads 4
  --max-turns 3
  --perception-model "${PERCEPTION_MODEL}"
  --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
  --captioner-config-yaml "${CAPTIONER_CONFIG_YAML}"
  --prompt-yaml "${PROMPT_YAML}"
  --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
  --gemini-api-backend legacy
  --signature-in-system-prompt
  --caption-placement task
)

echo "G3 GPT-5.4 planner best-candidate inference started at $(date -u +'%Y-%m-%dT%H:%M:%SZ')" | tee -a "${LOG_FILE}"
echo "compiled_program=${COMPILED_PROGRAM}" | tee -a "${LOG_FILE}"
echo "output_dir=${INFERENCE_OUTPUT_DIR}" | tee -a "${LOG_FILE}"
exec "${cmd[@]}" "$@" >> "${LOG_FILE}" 2>&1
