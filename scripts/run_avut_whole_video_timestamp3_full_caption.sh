#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This preserves all 1,734 benchmark questions while pointing the 72 formerly
# oversized recordings at existing <=18 MiB derivatives to avoid known 413s.
INPUT_JSONL="${AVUT_INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/avut/mllm_instruct_opt/avut_gemini2.5flash_direct_safe_duration_sorted_input/avut_full1734_duration_ascending_safe_videos.jsonl}"
OUTPUT_DIR="${AVUT_WHOLE_CAPTION_DIR:-/mnt/ceph_rbd/data/avqa_project/avut/avut_caption_gemini25flash_whole_video_timestamp3}"

exec "${SCRIPT_DIR}/run_full_caption_cache_common.sh" whole_timestamp3 "${INPUT_JSONL}" "${OUTPUT_DIR}"
