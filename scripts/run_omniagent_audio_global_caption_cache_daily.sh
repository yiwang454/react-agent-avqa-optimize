#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_ENV_FILE="${ROOT_ENV_FILE:-${REPO_DIR}/.env}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
TRACE_JSONL="${TRACE_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/OmniAgent_repeat1/output_test.jsonl}"
INPUT_JSONL="${INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_audio_global_caption_cache_omniagent_repeat1_completed}"
PROMPT_YAML="${PROMPT_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/omniagent_audio_global_caption_prompt.yaml}"
PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/config_omniagent_gemini25_flash_perception.yaml}"
MAX_WORKERS="${AUDIO_GLOBAL_CAPTION_MAX_WORKERS:-6}"

for required_path in \
  "${ROOT_ENV_FILE}" "${PYTHON_BIN}" "${TRACE_JSONL}" "${INPUT_JSONL}" \
  "${PROMPT_YAML}" "${PERCEPTION_CONFIG_YAML}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 2
  fi
done

set -a
source "${ROOT_ENV_FILE}"
set +a
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in ${ROOT_ENV_FILE} or the environment.}"

mkdir -p "${OUTPUT_DIR}"

COMMAND=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/build_omniagent_audio_global_caption_cache.py"
  --trace-jsonl "${TRACE_JSONL}"
  --input-jsonl "${INPUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --prompt-yaml "${PROMPT_YAML}"
  --prompt-key OMNIAGENT_AUDIO_GLOBAL_CAPTION_PROMPT
  --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
  --max-workers "${MAX_WORKERS}"
)

echo "Trace: ${TRACE_JSONL}"
echo "Input: ${INPUT_JSONL}"
echo "Output: ${OUTPUT_DIR}"
echo "Prompt YAML: ${PROMPT_YAML}"
echo "Perception config YAML: ${PERCEPTION_CONFIG_YAML}"
echo "Concurrent requests: ${MAX_WORKERS}"
printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/run.log"
