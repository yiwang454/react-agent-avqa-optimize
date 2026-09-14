#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

cd "${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"

PLANNER_CONFIG_YAML="${PLANNER_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
REFLECTION_CONFIG_YAML="${REFLECTION_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
GEPA_CONFIG_YAML="${GEPA_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/gepa_free_react_planner_2500.yaml}"
GEMINI_CONFIG_YAML="${GEMINI_CONFIG_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml}"
PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${GEMINI_CONFIG_YAML}}"
CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML_OVERRIDE:-${GEMINI_CONFIG_YAML}}"
PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_free_react_caption_in_task.yaml}"
INPUT_JSONL="${INPUT_JSONL:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_v3.jsonl}"
TRAINSET_JSONL="/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedTrain125.jsonl"
VALSET_JSONL="/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_cuts_selectedVal125.jsonl"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_caption_cache_v8_gemini2.5flash_seed27}"

EXPECTED_COUNT="${EXPECTED_COUNT:-1197}"
EXPECTED_TRAIN_COUNT=125
EXPECTED_VAL_COUNT=125
MAX_TURNS=4
GEPA_SEED="${GEPA_SEED:-18}"
GEPA_CALL_BUDGET=2500
FINAL_EVAL_NUM_THREADS="${FINAL_EVAL_NUM_THREADS:-4}"
FINAL_EVAL_BATCH_SIZE="${FINAL_EVAL_BATCH_SIZE:-${FINAL_EVAL_NUM_THREADS}}"

GEPA_RUN_DIR="${GEPA_RUN_DIR_OVERRIDE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_gepa_planner_maxturns4_seed${GEPA_SEED}_calls${GEPA_CALL_BUDGET}}"
export GEPA_RUN_DIR

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"
LOG_FILE="${LOG_FILE_OVERRIDE:-${GEPA_RUN_DIR}/${SCRIPT_NAME}.log}"

mkdir -p "${GEPA_RUN_DIR}"
mkdir -p "$(dirname "${LOG_FILE}")"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Log file: ${LOG_FILE}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi

: "${ELM_API_KEY:?Set ELM_API_KEY for the GPT-4.1 planner and reflection model.}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY for live Gemini captioning and perception.}"

export PLANNER_API_KEY="${ELM_API_KEY}"
unset DEEPSEEK_API_KEY DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE

export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="${GEMINI_API_BACKEND:-legacy}"
export GEMINI_RESPONSE_ERROR_SENSITIVE="false"

case "${GEMINI_API_BACKEND}" in
  legacy|dspy) ;;
  *)
    echo "GEMINI_API_BACKEND must be either legacy or dspy." >&2
    exit 2
    ;;
esac

for required_file in \
  "${PLANNER_CONFIG_YAML}" \
  "${REFLECTION_CONFIG_YAML}" \
  "${GEPA_CONFIG_YAML}" \
  "${PERCEPTION_CONFIG_YAML}" \
  "${CAPTIONER_CONFIG_YAML}" \
  "${PROMPT_YAML}" \
  "${INPUT_JSONL}" \
  "${TRAINSET_JSONL}" \
  "${VALSET_JSONL}" \
  "${CAPTION_CACHE_DIR}/manifest.json"; do
  if [ ! -f "${required_file}" ]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

count_rows() {
  awk 'NF {count++} END {print count+0}' "$1"
}

actual_count="$(count_rows "${INPUT_JSONL}")"
actual_train_count="$(count_rows "${TRAINSET_JSONL}")"
actual_val_count="$(count_rows "${VALSET_JSONL}")"
if [ "${actual_count}" -ne "${EXPECTED_COUNT}" ]; then
  echo "Input row count mismatch: expected ${EXPECTED_COUNT}, found ${actual_count} in ${INPUT_JSONL}" >&2
  exit 2
fi
if [ "${actual_train_count}" -ne "${EXPECTED_TRAIN_COUNT}" ]; then
  echo "Train row count mismatch: expected ${EXPECTED_TRAIN_COUNT}, found ${actual_train_count} in ${TRAINSET_JSONL}" >&2
  exit 2
fi
if [ "${actual_val_count}" -ne "${EXPECTED_VAL_COUNT}" ]; then
  echo "Validation row count mismatch: expected ${EXPECTED_VAL_COUNT}, found ${actual_val_count} in ${VALSET_JSONL}" >&2
  exit 2
fi
if [ "${FINAL_EVAL_NUM_THREADS}" -lt 1 ] || [ "${FINAL_EVAL_BATCH_SIZE}" -lt 1 ]; then
  echo "FINAL_EVAL_NUM_THREADS and FINAL_EVAL_BATCH_SIZE must be >= 1." >&2
  exit 2
fi

print_run_info() {
  echo "Workflow: GEPA training followed by full-v3 inference"
  echo "Optimize target: planner.workflow_prompt"
  echo "Planner: elm_gpt/gpt-4.1; reflection: elm_gpt/gpt-4.1"
  echo "Perception: live Gemini"
  echo "Captioning: first ask_caption uses the exact Gemini cache; later ask_caption calls use live Gemini."
  echo "Gemini API backend: ${GEMINI_API_BACKEND}"
  echo "Tool-call budget per rollout: ${MAX_TURNS} calls (including the required first ask_caption call)"
  echo "GEPA seed: ${GEPA_SEED}"
  echo "GEPA metric-call budget: ${GEPA_CALL_BUDGET}"
  echo "GEPA parameters: reflection_minibatch_size=16; candidate_selection_strategy=pareto; max_merge_invocations=5"
  echo "Training set: ${TRAINSET_JSONL} (${actual_train_count} rows; full Train125)"
  echo "Validation set: ${VALSET_JSONL} (${actual_val_count} rows)"
  echo "Final inference input: ${INPUT_JSONL} (${actual_count} rows)"
  echo "Prompt: ${PROMPT_YAML}"
  echo "Caption cache: ${CAPTION_CACHE_DIR}"
  echo "GEPA config: ${GEPA_CONFIG_YAML}"
  echo "Run directory: ${GEPA_RUN_DIR}"
  echo "Final output: ${GEPA_RUN_DIR}/output_test.jsonl"
  echo "Log file: ${LOG_FILE}"
}

cmd=(
  "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
  --input-jsonl "${INPUT_JSONL}"
  --max-turns "${MAX_TURNS}"
  --planner-config-yaml "${PLANNER_CONFIG_YAML}"
  --gepa-reflection-config-yaml "${REFLECTION_CONFIG_YAML}"
  --gepa-config "${GEPA_CONFIG_YAML}"
  --gepa-seed "${GEPA_SEED}"
  --perception-model "${PERCEPTION_MODEL}"
  --perception-config-yaml "${PERCEPTION_CONFIG_YAML}"
  --captioner-config-yaml "${CAPTIONER_CONFIG_YAML}"
  --prompt-yaml "${PROMPT_YAML}"
  --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
  --caption-cache-dir "${CAPTION_CACHE_DIR}"
  --caption-cache-scope first_call_only
  --ignore-audio-caption-dir
  --gemini-api-backend "${GEMINI_API_BACKEND}"
  --signature-in-system-prompt
  --caption-placement task
  --final-eval-num-threads "${FINAL_EVAL_NUM_THREADS}"
  --final-eval-batch-size "${FINAL_EVAL_BATCH_SIZE}"
  --print-config
)

if [ "${DRY_RUN:-false}" = "true" ]; then
  echo "Dry run: no optimization or inference request will be sent."
  print_run_info
  printf 'Command:'
  printf ' %q' "${cmd[@]}" "$@"
  printf '\n'
  exit 0
fi

echo "Follow live output with: tail -f ${LOG_FILE}"

echo "Started: $(date -Is)"
print_run_info
"${cmd[@]}" "$@"
echo "Finished: $(date -Is)"
