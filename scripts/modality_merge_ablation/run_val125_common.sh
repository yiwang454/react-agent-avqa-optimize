#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 B0|B1|B2|B3|B4|B5|B5-FLEX|B6|B7|B8 [runner args...]" >&2
  exit 2
fi

ABLATION_ID="${1^^}"
shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"
if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

case "${ABLATION_ID}" in
  B0)
    PROMPT_NAME="b0_omniagent_cqm.yaml"
    TOOLS="audio_global_caption,audio_qa,video_global_qa,video_clip_qa,video_metadata"
    CACHE_KIND="audio"
    ;;
  B1)
    PROMPT_NAME="b1_no_crosscheck_cqm.yaml"
    TOOLS="audio_global_caption,audio_qa,video_global_qa,video_clip_qa,video_metadata"
    CACHE_KIND="audio"
    ;;
  B2)
    PROMPT_NAME="b2_new_workflow_cqm.yaml"
    TOOLS="audio_global_caption,audio_qa,video_global_qa,video_clip_qa,video_metadata"
    CACHE_KIND="audio"
    ;;
  B3)
    PROMPT_NAME="b3_omni_caption_specialized_qm.yaml"
    TOOLS="ask_caption,audio_qa,video_global_qa,video_clip_qa,video_metadata"
    CACHE_KIND="omni"
    ;;
  B4)
    PROMPT_NAME="b4_omni_caption_specialized_qm_caption_first.yaml"
    TOOLS="ask_caption,audio_qa,video_global_qa,video_clip_qa,video_metadata"
    CACHE_KIND="omni"
    ;;
  B5)
    PROMPT_NAME="b5_omni_caption_global_clip_qm_caption_first.yaml"
    TOOLS="ask_caption,ask_perception,omni_clip_perception,video_metadata"
    CACHE_KIND="omni"
    ;;
  B5-FLEX|B5_FLEX|B5FLEX)
    ABLATION_ID="B5-FLEX"
    PROMPT_NAME="b5_flex_omni_caption_global_clip_qm.yaml"
    TOOLS="ask_caption,ask_perception,omni_clip_perception,video_metadata"
    CACHE_KIND="omni"
    ;;
  B6)
    PROMPT_NAME="b6_omni_caption_global_clip_q_no_metadata_caption_first.yaml"
    TOOLS="ask_caption,ask_perception,omni_clip_perception"
    CACHE_KIND="omni"
    ;;
  B7)
    PROMPT_NAME="b7_omni_caption_global_qm_no_clip_caption_first.yaml"
    TOOLS="ask_caption,ask_perception,video_metadata"
    CACHE_KIND="omni"
    ;;
  B8)
    PROMPT_NAME="b8_omni_caption_global_q_no_clip_no_metadata_caption_first.yaml"
    TOOLS="ask_caption,ask_perception"
    CACHE_KIND="omni"
    ;;
  *)
    echo "Unknown ablation ID: ${ABLATION_ID}" >&2
    exit 2
    ;;
esac

PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
INPUT_JSONL="${INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedVal125.jsonl}"
AUDIO_CAPTION_CACHE="${AUDIO_CAPTION_CACHE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_audio_global_caption_cache_omniagent_repeat1_completed}"
OMNI_CAPTION_CACHE="${OMNI_CAPTION_CACHE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_caption_cache_v8_gemini_from3repeats}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR_OVERRIDE:-$([ "${CACHE_KIND}" = "audio" ] && echo "${AUDIO_CAPTION_CACHE}" || echo "${OMNI_CAPTION_CACHE}")}"
PROMPT_YAML="${REPO_DIR}/DSPy/dspy_avqa/yamls/modality_merge_ablation/${PROMPT_NAME}"
PLANNER_CONFIG_YAML="${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_o3_medium.yaml"
PERCEPTION_CONFIG_YAML="${REPO_DIR}/DSPy/dspy_avqa/yamls/config_omniagent_gemini25_flash_perception.yaml"
SAMPLE_IDS="${SAMPLE_IDS:-}"
if [ -n "${SAMPLE_IDS}" ]; then
  DEFAULT_OUTPUT_ROOT="/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_smoke"
  SUMMARY_EXPECTED_COUNT="$(tr ',' '\n' <<<"${SAMPLE_IDS}" | awk 'NF {count++} END {print count+0}')"
else
  DEFAULT_OUTPUT_ROOT="/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_selectedVal125"
  SUMMARY_EXPECTED_COUNT="${EXPECTED_COUNT:-125}"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-${DEFAULT_OUTPUT_ROOT}}"
OUTPUT_DIR="${OUTPUT_DIR_OVERRIDE:-${OUTPUT_ROOT}/${ABLATION_ID,,}}"
OUTPUT_JSONL="${OUTPUT_JSONL_OVERRIDE:-${OUTPUT_DIR}/output_test.jsonl}"
MAX_TURNS="${MAX_TURNS:-6}"
INFERENCE_NUM_THREADS="${INFERENCE_NUM_THREADS:-4}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-${INFERENCE_NUM_THREADS}}"
EXPECTED_COUNT="${EXPECTED_COUNT:-125}"
LOG_FILE="${LOG_FILE_OVERRIDE:-${OUTPUT_DIR}/run.log}"

for required in "${PYTHON_BIN}" "${INPUT_JSONL}" "${PROMPT_YAML}" "${PLANNER_CONFIG_YAML}" "${PERCEPTION_CONFIG_YAML}" "${CAPTION_CACHE_DIR}/manifest.json"; do
  if [ ! -e "${required}" ]; then
    echo "Required path not found: ${required}" >&2
    exit 2
  fi
done
actual_count="$(awk 'NF {count++} END {print count+0}' "${INPUT_JSONL}")"
if [ "${actual_count}" -ne "${EXPECTED_COUNT}" ]; then
  echo "Input row count mismatch: expected ${EXPECTED_COUNT}, found ${actual_count}" >&2
  exit 2
fi
if [ "${MAX_TURNS}" -ne 6 ]; then
  echo "This controlled ablation fixes MAX_TURNS=6; got ${MAX_TURNS}." >&2
  exit 2
fi

: "${ELM_API_KEY:?Set ELM_API_KEY for the o3 planner.}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY for Gemini 2.5 Flash tools.}"
export PLANNER_API_KEY="${ELM_API_KEY}"
unset DEEPSEEK_API_KEY DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export DSPY_AVQA_ALLOWED_TOOLS="${TOOLS}"
export GEMINI_API_BACKEND="legacy"
export GEMINI_MODEL="gemini-2.5-flash"
export CAPTIONER_GEMINI_MODEL="gemini-2.5-flash"
export GEMINI_RESPONSE_ERROR_SENSITIVE="${GEMINI_RESPONSE_ERROR_SENSITIVE:-false}"

cmd=(
  "${PYTHON_BIN}" DSPy/avqa_dspy_impl.py
  --input-jsonl "${INPUT_JSONL}"
  --output-jsonl "${OUTPUT_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --max-turns "${MAX_TURNS}"
  --inference-num-threads "${INFERENCE_NUM_THREADS}"
  --inference-batch-size "${INFERENCE_BATCH_SIZE}"
  --planner-config-yaml "${PLANNER_CONFIG_YAML}"
  --perception-model gemini
  --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
  --prompt-yaml "${PROMPT_YAML}"
  --allowed-tools "${TOOLS}"
  --caption-cache-dir "${CAPTION_CACHE_DIR}"
  --caption-cache-scope all
  --ignore-audio-caption-dir
  --gemini-api-backend "${GEMINI_API_BACKEND}"
  --signature-in-system-prompt
  --caption-placement conversation_state
  --print-config
)
if [ -n "${SAMPLE_IDS}" ]; then
  cmd+=(--sample-ids "${SAMPLE_IDS}")
fi

print_run_info() {
  echo "Ablation: ${ABLATION_ID}"
  echo "Input: ${INPUT_JSONL} (${actual_count} rows)"
  echo "Prompt: ${PROMPT_YAML}"
  echo "Enabled tools: ${TOOLS}"
  echo "Caption cache (${CACHE_KIND}): ${CAPTION_CACHE_DIR}"
  echo "Planner: elm_gpt/o3, reasoning=medium, temperature=0, seed=1234"
  echo "Perception: Gemini 2.5 Flash with OmniAgent-compatible sampling config"
  echo "Budget: ${MAX_TURNS} valid tool calls; forced final is outside the budget"
  echo "History compression: disabled; optimization guidance: disabled"
  echo "Checkpoint/resume: ${OUTPUT_DIR}/*.json"
  echo "Selected sample IDs: ${SAMPLE_IDS:-all Val125}"
  echo "Output: ${OUTPUT_JSONL}"
}

cd "${REPO_DIR}"
mkdir -p "${OUTPUT_DIR}"
if [ "${DRY_RUN:-false}" = "true" ]; then
  print_run_info
  printf 'Command:'
  printf ' %q' "${cmd[@]}" "$@"
  printf '\n'
  exit 0
fi

{
  echo "Started: $(date -Is)"
  print_run_info
  "${cmd[@]}" "$@"
  "${PYTHON_BIN}" scripts/summarize_modality_merge_ablation.py \
    "${OUTPUT_JSONL}" --expected-count "${SUMMARY_EXPECTED_COUNT}" \
    --output-json "${OUTPUT_DIR}/metrics_summary.json"
  echo "Finished: $(date -Is)"
} 2>&1 | tee -a "${LOG_FILE}"
