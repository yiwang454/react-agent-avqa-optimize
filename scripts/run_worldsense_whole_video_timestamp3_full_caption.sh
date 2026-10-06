#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_JSONL="${WORLDSENSE_INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/WorldSense/worldsense_test_cut_old.jsonl}"
OUTPUT_DIR="${WORLDSENSE_WHOLE_CAPTION_DIR:-/mnt/ceph_rbd/data/avqa_project/WorldSense/worldsense_caption_gemini25flash_whole_video_timestamp3}"

exec "${SCRIPT_DIR}/run_full_caption_cache_common.sh" whole_timestamp3 "${INPUT_JSONL}" "${OUTPUT_DIR}"
