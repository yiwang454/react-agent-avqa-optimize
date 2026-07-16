#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/env_files/.env_dspy_react_agent_daily_qwen_v2}"

if [ ! -f "${ENV_FILE}" ]; then
  echo "Missing env file: ${ENV_FILE}" >&2
  exit 1
fi

set -a
source "${ENV_FILE}"
set +a

export PROJECT_DIR="${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"
cd "${PROJECT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ] && [ -n "${ENV_PREFIX:-}" ] && [ -x "${ENV_PREFIX}/bin/python" ]; then
  PYTHON_BIN="${ENV_PREFIX}/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"

# Match the requested v5 experiment while making the DeepSeek planner settings
# independent of any values inherited from ENV_FILE or the calling shell.
export PLANNER_PROVIDER="deepseek"
export PLANNER_TEMPERATURE="${PLANNER_TEMPERATURE_OVERRIDE:-0.0}"
export PLANNER_TOP_P="${PLANNER_TOP_P_OVERRIDE:-1.0}"
export PLANNER_THINKING_MODE="${PLANNER_THINKING_MODE_OVERRIDE:-disabled}"
export DEEPSEEK_SEED="${DEEPSEEK_SEED_OVERRIDE:-7}"
# reasoning_effort is only sent by the ELM-GPT planner path.  Keep it absent
# here (rather than sending the string "none") for this DeepSeek smoke test.
unset PLANNER_REASONING_EFFORT PLANNER_SEED

# GEPA reflects with a copy of the DeepSeek planner, retaining the original
# temp0p7 experiment's reflection temperature and fixed DeepSeek seed.
export GEPA_REFLECTION_TEMPERATURE="${GEPA_REFLECTION_TEMPERATURE_OVERRIDE:-0.7}"
unset GEPA_REFLECTION_MODEL GEPA_REFLECTION_REASONING_EFFORT

export QWEN_SEED="${QWEN_SEED_OVERRIDE:-1234}"
export QWEN_BASE_URL="${QWEN_BASE_URL_OVERRIDE:-http://10.62.186.52:8000/v1}"
export PERCEPTION_MODEL="${GEPA_PERCEPTION_MODEL:-qwen}"
export PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/config_localqwen_api_instruct.yaml}"
export PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${PROJECT_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v5.yaml}"
export DSPY_AVQA_ALLOWED_TOOLS="ask_perception"
export AUDIO_CAPTION_DIR="${GEPA_AUDIO_CAPTION_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_captioner_seed27_gemini-2.5-flash_QA_PROMPT_OMNI_CAPTIONER_TIMESTAMP3_0.0/}"

INPUT_JSONL="${GEPA_INPUT_JSONL:-${INPUT_JSONL}}"
TRAINSET_JSONL="${GEPA_TRAINSET_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedTrain125.jsonl}"
VALSET_JSONL="${GEPA_VALSET_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedVal125.jsonl}"
MAX_TURNS="${GEPA_MAX_TURNS:-4}"
DEBUG_LIMIT="${GEPA_DEBUG_LIMIT:-2}"
GEPA_SEED="${GEPA_SEED:-0}"
GEPA_MAX_METRIC_CALLS="${GEPA_SMOKE_MAX_METRIC_CALLS:-8}"
BASE_OUTPUT_DIR="${GEPA_BASE_ENV_OUTPUT_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_gepa_smoke_deepseek_qwen_v5_gemini_caption_temp0p7}"
RUN_DIR="${GEPA_RUN_DIR:-${BASE_OUTPUT_DIR}_qwen_seed_${QWEN_SEED}_deepseek_seed_${DEEPSEEK_SEED}_gepa_seed_${GEPA_SEED}}"
mkdir -p "${RUN_DIR}"

GEPA_CONFIG="${RUN_DIR}/gepa_smoke_config.json"
cat > "${GEPA_CONFIG}" <<JSON
{
  "auto": null,
  "max_full_evals": 1,
  "max_metric_calls": ${GEPA_MAX_METRIC_CALLS},
  "reflection_minibatch_size": 1,
  "candidate_selection_strategy": "current_best",
  "skip_perfect_score": false,
  "use_merge": false,
  "max_merge_invocations": 0,
  "num_threads": 1,
  "seed": ${GEPA_SEED},
  "log_dir": "${RUN_DIR}/gepa_logs",
  "reflective_dataset_save_interval": 50,
  "track_stats": true,
  "track_best_outputs": true
}
JSON

echo "GEPA smoke test: v5 + precomputed Gemini caption + Qwen perception + DeepSeek planner"
echo "RUN_DIR=${RUN_DIR}"
echo "PLANNER_PROVIDER=${PLANNER_PROVIDER}; DEEPSEEK_MODEL=${DEEPSEEK_MODEL:-}"
echo "PLANNER_TEMPERATURE=${PLANNER_TEMPERATURE}; PLANNER_REASONING_EFFORT=<unset>"
echo "PLANNER_THINKING_MODE=${PLANNER_THINKING_MODE}; DEEPSEEK_SEED=${DEEPSEEK_SEED}; PLANNER_TOP_P=${PLANNER_TOP_P}"
echo "GEPA_REFLECTION_TEMPERATURE=${GEPA_REFLECTION_TEMPERATURE}; GEPA_REFLECTION_REASONING_EFFORT=<unset>"
echo "QWEN_SEED=${QWEN_SEED}; QWEN_BASE_URL=${QWEN_BASE_URL}"
echo "PROMPT_YAML=${PROMPT_YAML}; PERCEPTION_CONFIG_YAML=${PERCEPTION_CONFIG_YAML}"
echo "AUDIO_CAPTION_DIR=${AUDIO_CAPTION_DIR}"
echo "MAX_TURNS=${MAX_TURNS}; DEBUG_LIMIT=${DEBUG_LIMIT}; max_metric_calls=${GEPA_MAX_METRIC_CALLS}; skip_perfect_score=false"

cmd=(
  "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
  --algorithm gepa
  --optimize-target planner.workflow_prompt
  --input-jsonl "${INPUT_JSONL}"
  --trainset-jsonl "${TRAINSET_JSONL}"
  --valset-jsonl "${VALSET_JSONL}"
  --audio-caption-dir "${AUDIO_CAPTION_DIR}"
  --output-program "${RUN_DIR}/compiled_gepa.json"
  --initial-program "${RUN_DIR}/initial_gepa_program.json"
  --metadata-json "${RUN_DIR}/compiled_gepa_metadata.json"
  --signature-search-json "${RUN_DIR}/compiled_gepa_signature_search.json"
  --optimizer-log-dir "${RUN_DIR}/optimizer_logs"
  --optimized-prompt-config-yaml "${RUN_DIR}/planner_workflow_prompt_smoke.yaml"
  --max-turns "${MAX_TURNS}"
  --perception-model "${PERCEPTION_MODEL}"
  --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
  --prompt-yaml "${PROMPT_YAML}"
  --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
  --signature-in-system-prompt
  --caption-placement task
  --gepa-config "${GEPA_CONFIG}"
  --debug
  --debug-limit "${DEBUG_LIMIT}"
)

cmd+=("$@")
"${cmd[@]}"

GEPA_OPTIMIZER_LOG="${RUN_DIR}/gepa_logs/gepa_optimizer.log"
CANDIDATE_LOG="${RUN_DIR}/planner_workflow_prompt_candidates.jsonl"
echo "GEPA optimizer log: ${GEPA_OPTIMIZER_LOG}"
if [ -f "${CANDIDATE_LOG}" ]; then
  CANDIDATE_COUNT="$(wc -l < "${CANDIDATE_LOG}")"
  echo "GEPA candidate count: ${CANDIDATE_COUNT} (${CANDIDATE_LOG})"
  if [ "${CANDIDATE_COUNT}" -le 1 ]; then
    echo "GEPA SMOKE DIAGNOSTIC: no new candidate completed a recordable evaluation." >&2
    echo "Inspect reflection/proposal tracebacks in: ${GEPA_OPTIMIZER_LOG}" >&2
  fi
else
  echo "GEPA SMOKE DIAGNOSTIC: candidate log was not created: ${CANDIDATE_LOG}" >&2
  echo "Inspect reflection/proposal tracebacks in: ${GEPA_OPTIMIZER_LOG}" >&2
fi
