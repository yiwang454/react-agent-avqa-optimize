#!/usr/bin/env bash
set -euo pipefail

# Re-run final inference twice for the completed G0 GPT-4.1 planner GEPA
# experiment.  This loads the selected program and does not run GEPA again.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_ENV_FILE="${ROOT_ENV_FILE:-${REPO_DIR}/.env}"

GEPA_RUN_DIR="${GEPA_RUN_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_v8GeminiCaptionInTask_gepa_planner_workflow_prompt_planner_gpt-4.1_seed_1234_gepa_seed18}"
COMPILED_PROGRAM="${COMPILED_PROGRAM:-${GEPA_RUN_DIR}/compiled_gepa.json}"

INPUT_JSONL="${INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl}"
DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
AUDIO_CAPTION_DIR="${AUDIO_CAPTION_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_captioner_seed27_gemini-2.5-flash_QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3_0.0/}"
PROMPT_YAML="${PROMPT_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task.yaml}"
PLANNER_CONFIG_YAML="${PLANNER_CONFIG_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
REFLECTION_CONFIG_YAML="${REFLECTION_CONFIG_YAML:-${PLANNER_CONFIG_YAML}}"
GEMINI_RUNTIME_CONFIG_YAML="${GEMINI_RUNTIME_CONFIG_YAML:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml}"
PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML:-${GEMINI_RUNTIME_CONFIG_YAML}}"
CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML:-${GEMINI_RUNTIME_CONFIG_YAML}}"

MAX_TURNS="${MAX_TURNS:-3}"
FINAL_EVAL_NUM_THREADS="${FINAL_EVAL_NUM_THREADS:-4}"

if [ ! -f "${ROOT_ENV_FILE}" ]; then
  echo "Missing env file: ${ROOT_ENV_FILE}" >&2
  exit 2
fi
if [ ! -f "${COMPILED_PROGRAM}" ]; then
  echo "Missing completed GEPA program: ${COMPILED_PROGRAM}" >&2
  exit 2
fi

set -a
source "${ROOT_ENV_FILE}"
set +a

: "${ELM_API_KEY:?ELM_API_KEY must be set in ${ROOT_ENV_FILE}}"
export PROJECT_DIR="${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"
export PLANNER_API_KEY="${ELM_API_KEY}"
export DEEPSEEK_API_KEY="${PLANNER_API_KEY}"
unset DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
export DSPY_AVQA_ALLOWED_TOOLS="${DSPY_AVQA_ALLOWED_TOOLS:-ask_caption,ask_perception}"
export GEMINI_API_BACKEND="legacy"
export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"

for required_path in \
  "${INPUT_JSONL}" \
  "${PROMPT_YAML}" \
  "${PLANNER_CONFIG_YAML}" \
  "${REFLECTION_CONFIG_YAML}" \
  "${PERCEPTION_CONFIG_YAML}" \
  "${CAPTIONER_CONFIG_YAML}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 2
  fi
done

run_repeat() {
  local repeat_tag="$1"
  shift

  local output_dir="${GEPA_RUN_DIR}/dailyomni_inference_${repeat_tag}"
  local log_file="${output_dir}/run.log"
  mkdir -p "${output_dir}"

  local archived_script="${output_dir}/$(basename "${BASH_SOURCE[0]}")"
  if [ ! -e "${archived_script}" ]; then
    cp -p "${BASH_SOURCE[0]}" "${archived_script}"
  fi

  local cmd=(
    python DSPy/avqa_dspy_optimize.py
    --algorithm gepa
    --inference-only
    --input-jsonl "${INPUT_JSONL}"
    --daily-omni-root "${DAILY_OMNI_ROOT}"
    --audio-caption-dir "${AUDIO_CAPTION_DIR}"
    --output-program "${COMPILED_PROGRAM}"
    --metadata-json "${output_dir}/inference_only_metadata.json"
    --final-eval-output-jsonl "${output_dir}/output_test.jsonl"
    --final-eval-output-dir "${output_dir}"
    --final-eval-num-threads "${FINAL_EVAL_NUM_THREADS}"
    --max-turns "${MAX_TURNS}"
    --planner-config-yaml "${PLANNER_CONFIG_YAML}"
    --gepa-reflection-config-yaml "${REFLECTION_CONFIG_YAML}"
    --perception-model "${PERCEPTION_MODEL}"
    --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
    --captioner-config-yaml "${CAPTIONER_CONFIG_YAML}"
    --prompt-yaml "${PROMPT_YAML}"
    --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
    --gemini-api-backend legacy
    --signature-in-system-prompt
    --caption-placement task
    --print-config
  )

  echo "Log file: ${log_file}"
  "${cmd[@]}" "$@" >> "${log_file}" 2>&1
}

cd "${PROJECT_DIR}"
run_repeat repeat2 "$@"
run_repeat repeat3 "$@"
