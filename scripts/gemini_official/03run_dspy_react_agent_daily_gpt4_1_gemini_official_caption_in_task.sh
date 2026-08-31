#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
LOG_FILE="${REPO_DIR}/scripts/logs/${SCRIPT_NAME%.sh}_repeat2.log"

ORIGINAL_SCRIPT="${REPO_DIR}/scripts/01_run_dspy_react_agent_daily_qwen_v8_GPT4.1_gemini_captionInTask.sh"
OFFICIAL_GEMINI_CONFIG_YAML="${OFFICIAL_GEMINI_CONFIG_YAML:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_official_cold.yaml}"

# Keep the original ReAct experiment settings, changing only the Gemini
# transport/configuration and isolating official-Vertex output artifacts.
export GEMINI_API_BACKEND_OVERRIDE="dspy"
export PERCEPTION_CONFIG_YAML_OVERRIDE="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${OFFICIAL_GEMINI_CONFIG_YAML}}"
export CAPTIONER_CONFIG_YAML_OVERRIDE="${CAPTIONER_CONFIG_YAML_OVERRIDE:-${OFFICIAL_GEMINI_CONFIG_YAML}}"
export GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-/mnt/ceph_rbd/workspace/avqa_project/avqa_reasoning_datasets/shared_scripts/decoupled-avqa-501420-31646efd16fd.json}"
export GEMINI_LOCAL_DATA_ROOT="${GEMINI_LOCAL_DATA_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
export GEMINI_GCS_DATA_ROOT="${GEMINI_GCS_DATA_ROOT:-gs://audiovideo_bucket-1/daily_omni}"

export OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_GPT_v8GeminiCaptionInTask_officialGemini_planner_gpt-4.1_seed_1234_repeat2}"

cd "${REPO_DIR}"

# echo "${SCRIPT_NAME}"
echo "${LOG_FILE}"
exec "${ORIGINAL_SCRIPT}" \
  --gemini-api-backend "${GEMINI_API_BACKEND_OVERRIDE}" \
  --perception-config-yaml "${PERCEPTION_CONFIG_YAML_OVERRIDE}" \
  --captioner-config-yaml "${CAPTIONER_CONFIG_YAML_OVERRIDE}" \
  --gemini-local-data-root "${GEMINI_LOCAL_DATA_ROOT}" \
  --gemini-gcs-data-root "${GEMINI_GCS_DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR_OVERRIDE}" \
  --output-jsonl "${OUTPUT_DIR_OVERRIDE}/output_test.jsonl" \
  "$@" >> "${LOG_FILE}" 2>&1
