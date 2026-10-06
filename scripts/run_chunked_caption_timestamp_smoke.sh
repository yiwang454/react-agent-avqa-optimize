#!/usr/bin/env bash
set -euo pipefail

# Compare the production local-offset timestamp strategy with the lightweight
# model-original strategy on three approximately five-minute WorldSense videos.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_ENV_FILE="${ROOT_ENV_FILE:-${REPO_DIR}/.env}"
PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml}"
PROMPT_YAML="${PROMPT_YAML:-${REPO_DIR}/DSPy/dspy_avqa/yamls/omni_caption_prompt.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/ceph_rbd/data/avqa_project/WorldSense/chunked_caption_timestamp_smoke_20261004}"
MAX_WORKERS="${MAX_WORKERS:-5}"
CHUNK_PROMPT_PROFILE="${CHUNK_PROMPT_PROFILE:-CHUNKED_CAPTION_PROMPT}"

VIDEO_PATHS=(
  "/mnt/ceph_rbd/data/avqa_project/WorldSense/videos/ayCJscCe.mp4"
  "/mnt/ceph_rbd/data/avqa_project/WorldSense/videos/NUhenCVt.mp4"
  "/mnt/ceph_rbd/data/avqa_project/WorldSense/videos/sKzAjbxS.mp4"
)

for required_path in "${ROOT_ENV_FILE}" "${PYTHON_BIN}" "${CAPTIONER_CONFIG_YAML}" "${PROMPT_YAML}" "${VIDEO_PATHS[@]}"; do
  if [ ! -e "${required_path}" ]; then
    echo "Missing required path: ${required_path}" >&2
    exit 2
  fi
done

set -a
source "${ROOT_ENV_FILE}"
set +a

mkdir -p "${OUTPUT_ROOT}/local_offset" "${OUTPUT_ROOT}/model_original"
COMMON_ARGS=(
  --captioner-config-yaml "${CAPTIONER_CONFIG_YAML}"
  --prompt-yaml "${PROMPT_YAML}"
  --chunk-prompt-profile "${CHUNK_PROMPT_PROFILE}"
  --chunk-seconds 60
  --min-tail-seconds 30
  --max-workers "${MAX_WORKERS}"
  --gemini-api-backend legacy
  --skip-question-cache
  --force
)
for video_path in "${VIDEO_PATHS[@]}"; do
  COMMON_ARGS+=(--video-path "${video_path}")
done

echo "Local-offset output: ${OUTPUT_ROOT}/local_offset"
local_status=0
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_chunked_caption_cache.py" \
  "${COMMON_ARGS[@]}" \
  --timestamp-mode local_offset \
  --output-dir "${OUTPUT_ROOT}/local_offset" \
  2>&1 | tee -a "${OUTPUT_ROOT}/local_offset/run.log" || local_status=$?

echo "Model-original output: ${OUTPUT_ROOT}/model_original"
original_status=0
"${PYTHON_BIN}" "${SCRIPT_DIR}/build_chunked_caption_cache.py" \
  "${COMMON_ARGS[@]}" \
  --timestamp-mode model_original \
  --output-dir "${OUTPUT_ROOT}/model_original" \
  2>&1 | tee -a "${OUTPUT_ROOT}/model_original/run.log" || original_status=$?

"${PYTHON_BIN}" "${SCRIPT_DIR}/compare_chunked_caption_smoke.py" \
  --local-output-dir "${OUTPUT_ROOT}/local_offset" \
  --original-output-dir "${OUTPUT_ROOT}/model_original" \
  --output-markdown "${OUTPUT_ROOT}/comparison.md" \
  --output-json "${OUTPUT_ROOT}/comparison.json"

echo "Comparison report: ${OUTPUT_ROOT}/comparison.md"
if [ "${local_status}" -ne 0 ] || [ "${original_status}" -ne 0 ]; then
  echo "Smoke generation exit codes: local_offset=${local_status}, model_original=${original_status}" >&2
  exit 1
fi
