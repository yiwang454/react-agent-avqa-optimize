#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export QWEN_SEED="${QWEN_SEED:-1234}"
export DEEPSEEK_SEED="${DEEPSEEK_SEED:-7}"
export DSPY_AVQA_CAPTION_CACHE_DIR="${DSPY_AVQA_CAPTION_CACHE_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_caption_cache_v8_gemini_repeat1_fallback_repeat2_repeat3}"
export OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_qwen_v8CachedGeminiCaptionInTask_qwen_seed_${QWEN_SEED}_deepseek_seed_${DEEPSEEK_SEED}}"
export AUDIO_CAPTION_DIR_OVERRIDE=""

if [ ! -f "${DSPY_AVQA_CAPTION_CACHE_DIR}/manifest.json" ]; then
  echo "Caption cache manifest not found: ${DSPY_AVQA_CAPTION_CACHE_DIR}/manifest.json" >&2
  exit 1
fi

exec "${SCRIPT_DIR}/run_dspy_react_agent_daily_qwen_v8_gemini_captionInTask_seed.sh" \
  --caption-cache-dir "${DSPY_AVQA_CAPTION_CACHE_DIR}" \
  --ignore-audio-caption-dir \
  "$@"
