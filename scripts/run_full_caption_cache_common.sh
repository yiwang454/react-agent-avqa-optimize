#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <split_avs_plan_a|whole_timestamp3> <input-jsonl> <output-dir>" >&2
  exit 2
fi

STRATEGY="$1"
INPUT_JSONL="$2"
OUTPUT_DIR="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_ENV_FILE="${ROOT_ENV_FILE:-${REPO_DIR}/.env}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
PROMPT_YAML="${PROMPT_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/omni_caption_prompt.yaml}"
CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/config_gemini2_5_flash_captioner.yaml}"
SEGMENT_MAX_WORKERS="${CAPTION_SEGMENT_MAX_WORKERS:-6}"
WHOLE_VIDEO_MAX_WORKERS="${CAPTION_WHOLE_VIDEO_MAX_WORKERS:-3}"

for required_path in "${ROOT_ENV_FILE}" "${PYTHON_BIN}" "${PROMPT_YAML}" "${CAPTIONER_CONFIG_YAML}" "${INPUT_JSONL}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 2
  fi
done

set -a
source "${ROOT_ENV_FILE}"
set +a

case "${STRATEGY}" in
  split_avs_plan_a)
    MAX_WORKERS="${SEGMENT_MAX_WORKERS}"
    COMMAND=(
      "${PYTHON_BIN}" "${SCRIPT_DIR}/build_chunked_caption_cache.py"
      --input-jsonl "${INPUT_JSONL}"
      --output-dir "${OUTPUT_DIR}"
      --prompt-yaml "${PROMPT_YAML}"
      --captioner-config-yaml "${CAPTIONER_CONFIG_YAML}"
      --chunk-prompt-profile CHUNKED_CAPTION_PROMPT_SPLIT_AVS
      --timestamp-mode local_offset
      --chunk-seconds 60
      --min-tail-seconds 30
      --max-workers "${MAX_WORKERS}"
      --gemini-api-backend legacy
    )
    ;;
  whole_timestamp3)
    MAX_WORKERS="${WHOLE_VIDEO_MAX_WORKERS}"
    COMMAND=(
      "${PYTHON_BIN}" "${SCRIPT_DIR}/build_whole_video_caption_cache.py"
      --input-jsonl "${INPUT_JSONL}"
      --output-dir "${OUTPUT_DIR}"
      --prompt-yaml "${PROMPT_YAML}"
      --prompt-key QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3
      --captioner-config-yaml "${CAPTIONER_CONFIG_YAML}"
      --max-workers "${MAX_WORKERS}"
      --gemini-api-backend legacy
    )
    ;;
  *)
    echo "Unsupported strategy: ${STRATEGY}" >&2
    exit 2
    ;;
esac

if [ "${CAPTION_LAUNCH_DRY_RUN:-0}" = "1" ]; then
  printf 'Command:'
  printf ' %q' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "${OUTPUT_DIR}"

echo "Strategy: ${STRATEGY}"
echo "Input: ${INPUT_JSONL}"
echo "Output: ${OUTPUT_DIR}"
echo "Concurrent requests: ${MAX_WORKERS}"
"${COMMAND[@]}" 2>&1 | tee -a "${OUTPUT_DIR}/run.log"
