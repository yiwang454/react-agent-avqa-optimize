#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_JSONL="${WORLDSENSE_INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/WorldSense/worldsense_test_cut_old.jsonl}"
OUTPUT_DIR="${WORLDSENSE_SPLIT_AVS_CAPTION_DIR:-/mnt/ceph_rbd/data/avqa_project/WorldSense/worldsense_caption_gemini25flash_split_avs_plan_a_timestamp5}"

exec "${SCRIPT_DIR}/run_full_caption_cache_common.sh" split_avs_plan_a "${INPUT_JSONL}" "${OUTPUT_DIR}"
