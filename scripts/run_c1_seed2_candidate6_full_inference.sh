#!/usr/bin/env bash
set -euo pipefail

# Full DailyOmni-v3 inference using the compiled artifact in the seed-2 GEPA run.
# This launcher never reruns GEPA.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

cd "${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
INPUT_JSONL="${INPUT_JSONL:-${DAILY_OMNI_ROOT}/daily_omni_cuts_v3.jsonl}"
GEPA_RUN_DIR="${GEPA_RUN_DIR:-${DAILY_OMNI_ROOT}/daily_omni_dspy_free_react_o3_gemini25flash_basic_tools_gepa_guidance_maxturns6_seed2_calls2500}"
COMPILED_PROGRAM="${COMPILED_PROGRAM:-${GEPA_RUN_DIR}/compiled_gepa.json}"
PROMPT_YAML="${PROMPT_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_free_react_guidance_caption_in_task.yaml}"
PLANNER_CONFIG_YAML="${PLANNER_CONFIG_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_o3_medium.yaml}"
REFLECTION_CONFIG_YAML="${REFLECTION_CONFIG_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_o3_high.yaml}"
GEMINI_CONFIG_YAML="${GEMINI_CONFIG_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/config_gemini25_flash_env.yaml}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-${DAILY_OMNI_ROOT}/daily_omni_caption_cache_v8_gemini_from3repeats}"
OUTPUT_DIR="${OUTPUT_DIR:-${DAILY_OMNI_ROOT}/daily_omni_dspy_free_react_o3_gemini25flash_guidance_seed2_candidate6_full_v3}"
FINAL_EVAL_NUM_THREADS="${FINAL_EVAL_NUM_THREADS:-4}"
MAX_TURNS="${MAX_TURNS:-6}"
EXPECTED_FINAL_COUNT="${EXPECTED_FINAL_COUNT:-1197}"

: "${ELM_API_KEY:?Set ELM_API_KEY for the o3 planner.}"
export PLANNER_API_KEY="${ELM_API_KEY}"
unset DEEPSEEK_API_KEY DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE

export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="${GEMINI_API_BACKEND:-legacy}"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"
export CAPTIONER_GEMINI_MODEL="${CAPTIONER_GEMINI_MODEL:-${GEMINI_MODEL}}"
export GEMINI_RESPONSE_ERROR_SENSITIVE="${GEMINI_RESPONSE_ERROR_SENSITIVE:-false}"

for required_path in \
  "${PYTHON_BIN}" \
  "${INPUT_JSONL}" \
  "${COMPILED_PROGRAM}" \
  "${PROMPT_YAML}" \
  "${PLANNER_CONFIG_YAML}" \
  "${REFLECTION_CONFIG_YAML}" \
  "${GEMINI_CONFIG_YAML}" \
  "${CAPTION_CACHE_DIR}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${LOG_FILE_OVERRIDE:-${OUTPUT_DIR}/run.log}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Inference-only full-v3 run: compiled artifact from GEPA seed 2"
echo "Compiled program: ${COMPILED_PROGRAM}"
echo "Optimize target: planner.optimization_guidance"
echo "Output directory: ${OUTPUT_DIR}"
echo "Final-eval threads: ${FINAL_EVAL_NUM_THREADS}"

"${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py \
  --algorithm gepa \
  --inference-only \
  --optimize-target planner.optimization_guidance \
  --input-jsonl "${INPUT_JSONL}" \
  --daily-omni-root "${DAILY_OMNI_ROOT}" \
  --output-program "${COMPILED_PROGRAM}" \
  --metadata-json "${OUTPUT_DIR}/inference_only_metadata.json" \
  --final-eval-output-jsonl "${OUTPUT_DIR}/output_test.jsonl" \
  --final-eval-output-dir "${OUTPUT_DIR}" \
  --final-eval-num-threads "${FINAL_EVAL_NUM_THREADS}" \
  --max-turns "${MAX_TURNS}" \
  --planner-config-yaml "${PLANNER_CONFIG_YAML}" \
  --gepa-reflection-config-yaml "${REFLECTION_CONFIG_YAML}" \
  --perception-model "${PERCEPTION_MODEL}" \
  --perception-config-yaml "${GEMINI_CONFIG_YAML}" \
  --captioner-config-yaml "${GEMINI_CONFIG_YAML}" \
  --prompt-yaml "${PROMPT_YAML}" \
  --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}" \
  --caption-cache-dir "${CAPTION_CACHE_DIR}" \
  --caption-cache-scope first_call_only \
  --ignore-audio-caption-dir \
  --gemini-api-backend "${GEMINI_API_BACKEND}" \
  --signature-in-system-prompt \
  --caption-placement task \
  --print-config \
  "$@"

"${PYTHON_BIN}" scripts/aggregate_inference_latency.py \
  "${OUTPUT_DIR}/output_test.jsonl" \
  --expected-count "${EXPECTED_FINAL_COUNT}" \
  --output-json "${OUTPUT_DIR}/latency_summary.json"
