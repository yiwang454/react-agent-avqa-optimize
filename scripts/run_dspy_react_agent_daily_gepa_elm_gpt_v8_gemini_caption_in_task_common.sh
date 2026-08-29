#!/usr/bin/env bash
set -euo pipefail

export PROJECT_DIR="."
COMMON_SCRIPT_PATH="./scripts/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh"
ROOT_ENV_FILE="./.env"
: "${GEPA_LAUNCHER_PATH:?GEPA_LAUNCHER_PATH must be set by the experiment wrapper}"
: "${GEPA_PLANNER_CONFIG_YAML:?GEPA_PLANNER_CONFIG_YAML must be set by the experiment wrapper}"
: "${GEPA_REFLECTION_CONFIG_YAML:?GEPA_REFLECTION_CONFIG_YAML must be set by the experiment wrapper}"
: "${GEPA_RUN_DIR:?GEPA_RUN_DIR must be set by the experiment wrapper}"
: "${GEPA_SEED:?GEPA_SEED must be set by the experiment wrapper}"

set -a
source "${ROOT_ENV_FILE}"
set +a

PYTHON_BIN="${ENV_PREFIX}/bin/python"
GEPA_CONFIG_YAML="./DSPy/dspy_avqa/yamls/gepa_g0_planner.yaml"
: "${ELM_API_KEY:?ELM_API_KEY must be set in .env}"
export PLANNER_API_KEY="${ELM_API_KEY}"
export DEEPSEEK_API_KEY="${PLANNER_API_KEY}"
unset DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE

export PROMPT_YAML="./DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_caption_in_task.yaml"
export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export PERCEPTION_CONFIG_YAML="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
export CAPTIONER_CONFIG_YAML="/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="legacy"
MAX_TURNS=3

export INPUT_JSONL="/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl"
AUDIO_CAPTION_DIR="/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_captioner_seed27_gemini-2.5-flash_QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3_0.0/"

## copy scripts and dependent scripts to output directory
archived_launcher="${GEPA_RUN_DIR}/$(basename "${GEPA_LAUNCHER_PATH}")"
archived_common="${GEPA_RUN_DIR}/run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh"
for source_and_target in \
  "${GEPA_LAUNCHER_PATH}|${archived_launcher}" \
  "${COMMON_SCRIPT_PATH}|${archived_common}"; do
  source_path="${source_and_target%%|*}"
  target_path="${source_and_target#*|}"
  if [ -e "${target_path}" ] && ! cmp -s "${source_path}" "${target_path}"; then
    printf 'Archived launcher differs from current script: %s\n' "${target_path}" >&2
    exit 2
  fi
done
## finished copy scripts

cmd=(
  "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
  --input-jsonl "${INPUT_JSONL}"
  --audio-caption-dir "${AUDIO_CAPTION_DIR}"
  --max-turns "${MAX_TURNS}"
  --planner-config-yaml "${GEPA_PLANNER_CONFIG_YAML}"
  --gepa-reflection-config-yaml "${GEPA_REFLECTION_CONFIG_YAML}"
  --gepa-config "${GEPA_CONFIG_YAML}"
  --gepa-seed "${GEPA_SEED}"
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
cmd+=("$@")
"${cmd[@]}"

if [ ! -e "${archived_launcher}" ]; then
  cp -p "${GEPA_LAUNCHER_PATH}" "${archived_launcher}"
fi
if [ ! -e "${archived_common}" ]; then
  cp -p "${COMMON_SCRIPT_PATH}" "${archived_common}"
fi
