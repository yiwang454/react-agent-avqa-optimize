#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_caption_cache_v8_gemini_repeat1_fallback_repeat2_repeat3}"
export GEPA_EXPERIMENT_NAME="${GEPA_EXPERIMENT_NAME:-g0_deepseek_qwen_cached_gemini_v8_caption_in_task_planner_gpt5_4_reflection}"
export GEPA_BASE_ENV_OUTPUT_DIR="${GEPA_BASE_ENV_OUTPUT_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_deepseek_qwen_cached_gemini_v8_gepa_${GEPA_EXPERIMENT_NAME}}"

if [ ! -f "${CAPTION_CACHE_DIR}/manifest.json" ]; then
  echo "Caption cache manifest not found: ${CAPTION_CACHE_DIR}/manifest.json" >&2
  exit 1
fi

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_gepa_g0_deepseek_qwen_gemini_v8_caption_in_task_planner_gpt5_4_reflection.sh" \
  --caption-cache-dir "${CAPTION_CACHE_DIR}" \
  "$@"
