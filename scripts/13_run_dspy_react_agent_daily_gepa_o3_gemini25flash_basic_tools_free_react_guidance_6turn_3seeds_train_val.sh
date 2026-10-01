#!/usr/bin/env bash
set -euo pipefail

# Script-11 Free-ReAct baseline with guidance-only GEPA. The workflow contract
# stays fixed; seeds 18/42/2 optimize planner.optimization_guidance in parallel
# on Train125/Val125. Full-v3 inference is intentionally not run here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

cd "${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
PLANNER_CONFIG_YAML="${PLANNER_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_o3_medium.yaml}"
REFLECTION_CONFIG_YAML="${REFLECTION_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_o3_high.yaml}"
GEMINI_CONFIG_YAML="${GEMINI_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/config_gemini25_flash_env.yaml}"
PERCEPTION_CONFIG_YAML="${PERCEPTION_CONFIG_YAML_OVERRIDE:-${GEMINI_CONFIG_YAML}}"
CAPTIONER_CONFIG_YAML="${CAPTIONER_CONFIG_YAML_OVERRIDE:-${GEMINI_CONFIG_YAML}}"
PROMPT_YAML="${PROMPT_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_free_react_guidance_caption_in_task.yaml}"
REFLECTION_TEMPLATE_YAML="${REFLECTION_TEMPLATE_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/DSPy/reflection_template.yaml}"

DAILY_OMNI_ROOT="${DAILY_OMNI_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
INPUT_JSONL="${INPUT_JSONL:-${DAILY_OMNI_ROOT}/daily_omni_cuts_v3.jsonl}"
TRAINSET_JSONL="${TRAINSET_JSONL:-${DAILY_OMNI_ROOT}/daily_omni_cuts_selectedTrain125.jsonl}"
VALSET_JSONL="${VALSET_JSONL:-${DAILY_OMNI_ROOT}/daily_omni_cuts_selectedVal125.jsonl}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-${DAILY_OMNI_ROOT}/daily_omni_caption_cache_v8_gemini_from3repeats}"
RUN_PREFIX="${GEPA_RUN_PREFIX_OVERRIDE:-${DAILY_OMNI_ROOT}/daily_omni_dspy_free_react_o3_gemini25flash_basic_tools_gepa_guidance_maxturns6}"

MAX_TURNS=6
GEPA_CALL_BUDGET=2500
GEPA_SEEDS=(18 42 2)

: "${ELM_API_KEY:?Set ELM_API_KEY for the o3 planner and reflection model.}"
export PLANNER_API_KEY="${ELM_API_KEY}"
unset DEEPSEEK_API_KEY DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE

export PERCEPTION_MODEL="gemini"
export CAPTIONER_MODEL="gemini"
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="${GEMINI_API_BACKEND:-legacy}"
export GEMINI_MODEL="${GEMINI_MODEL:-gemini-2.5-flash}"
export CAPTIONER_GEMINI_MODEL="${CAPTIONER_GEMINI_MODEL:-${GEMINI_MODEL}}"
export GEMINI_RESPONSE_ERROR_SENSITIVE="${GEMINI_RESPONSE_ERROR_SENSITIVE:-false}"

common_runtime_args=(
  --input-jsonl "${INPUT_JSONL}"
  --max-turns "${MAX_TURNS}"
  --planner-config-yaml "${PLANNER_CONFIG_YAML}"
  --gepa-reflection-config-yaml "${REFLECTION_CONFIG_YAML}"
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
  --print-config
)

echo "Free-ReAct guidance-only GEPA: parallel seeds=${GEPA_SEEDS[*]}"
echo "Planner: o3 medium; reflection: o3 high; max turns: ${MAX_TURNS}"
echo "Optimize target: planner.optimization_guidance"

pids=()
run_dirs=()

cleanup_children() {
  trap - EXIT INT TERM
  if [ "${#pids[@]}" -gt 0 ]; then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
}

trap cleanup_children EXIT
trap 'cleanup_children; exit 130' INT
trap 'cleanup_children; exit 143' TERM

for seed in "${GEPA_SEEDS[@]}"; do
  run_dir="${RUN_PREFIX}_seed${seed}_calls${GEPA_CALL_BUDGET}"
  mkdir -p "${run_dir}"

  cmd=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
    --algorithm gepa
    --optimization-train-only
    --trainset-jsonl "${TRAINSET_JSONL}"
    --valset-jsonl "${VALSET_JSONL}"
    --daily-omni-root "${DAILY_OMNI_ROOT}"
    --output-program "${run_dir}/compiled_gepa.json"
    --initial-program "${run_dir}/initial_gepa_program.json"
    --metadata-json "${run_dir}/compiled_gepa_metadata.json"
    --signature-search-json "${run_dir}/compiled_gepa_signature_search.json"
    --trajectory-jsonl "${run_dir}/optimized_trainset_trajectories.jsonl"
    --optimizer-log-dir "${run_dir}/optimizer_logs"
    --optimized-prompt-config-yaml "${run_dir}/planner_optimization_guidance_GEPA_daily_qa_prompt_v8_free_react_guidance_caption_in_task.yaml"
    --optimize-target planner.optimization_guidance
    --gepa-seed "${seed}"
    --gepa-max-metric-calls "${GEPA_CALL_BUDGET}"
    --gepa-reflection-minibatch-size 16
    --gepa-candidate-selection-strategy pareto
    --gepa-max-merge-invocations 5
    --gepa-reflection-template-yaml "${REFLECTION_TEMPLATE_YAML}"
    --gepa-reflection-template-version auto
    --gepa-log-dir "${run_dir}/gepa_logs"
    --gepa-track-stats
    --gepa-track-best-outputs
    "${common_runtime_args[@]}"
  )

  (
    echo "Seed ${seed} started: $(date -Is)"
    "${cmd[@]}" "$@"
    echo "Seed ${seed} finished: $(date -Is)"
  ) > >(tee -a "${run_dir}/run.log") 2>&1 &
  pids+=("$!")
  run_dirs+=("${run_dir}")
done

failures=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "Completed: ${run_dirs[index]}"
  else
    echo "Failed: ${run_dirs[index]} (see run.log)" >&2
    failures=$((failures + 1))
  fi
done
trap - EXIT INT TERM

if [ "${failures}" -ne 0 ]; then
  echo "${failures} of ${#GEPA_SEEDS[@]} GEPA searches failed." >&2
  exit 1
fi
echo "All ${#GEPA_SEEDS[@]} guidance-only GEPA searches completed."
