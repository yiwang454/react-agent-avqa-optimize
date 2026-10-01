#!/usr/bin/env bash
set -euo pipefail

# Run the best guidance-only GEPA programs from seeds 42 and 2 on full
# DailyOmni-v3. The two inference jobs run concurrently; each job uses exactly
# three final-evaluation workers.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${ENV_FILE:-${REPO_DIR}/.env}"

if [ -f "${ENV_FILE}" ]; then
  set -a
  source "${ENV_FILE}"
  set +a
fi

cd "${PROJECT_DIR_OVERRIDE:-${REPO_DIR}}"

if [ "$#" -ne 0 ]; then
  echo "This launcher does not accept positional arguments; use the documented environment overrides." >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-/mnt/ceph_rbd/applications/anaconda3/envs/react-avqa-dspy/bin/python}"
DATA_ROOT="${DATA_ROOT:-/mnt/ceph_rbd/data/avqa_project/daily_omni}"
RUN_PREFIX="${GEPA_RUN_PREFIX_OVERRIDE:-${DATA_ROOT}/daily_omni_dspy_free_react_gpt4_1_gemini_first_cached_then_live_caption_gepa_guidance_maxturns4}"

INPUT_JSONL="${INPUT_JSONL:-${DATA_ROOT}/daily_omni_cuts_v3.jsonl}"
TRAINSET_JSONL="${TRAINSET_JSONL:-${DATA_ROOT}/daily_omni_cuts_selectedTrain125.jsonl}"
VALSET_JSONL="${VALSET_JSONL:-${DATA_ROOT}/daily_omni_cuts_selectedVal125.jsonl}"
CAPTION_CACHE_DIR="${CAPTION_CACHE_DIR:-${DATA_ROOT}/daily_omni_caption_cache_v8_gemini2.5flash_seed27}"
PLANNER_CONFIG_YAML="${PLANNER_CONFIG_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/reasoner_elm_gpt4_1_none.yaml}"
GEMINI_CONFIG_YAML="${GEMINI_CONFIG_YAML_OVERRIDE:-/mnt/ceph_rbd/workspace/avqa_project/demos/yamls/gemini_qa/daily_125_gemini2.5_cold_captioner.yaml}"
BASE_PROMPT_YAML="${BASE_PROMPT_YAML_OVERRIDE:-${REPO_DIR}/DSPy/dspy_avqa/yamls/daily_qa_prompt_v8_free_react_guidance_caption_in_task.yaml}"

EXPECTED_COUNT=1197
EXPECTED_TRAIN_COUNT=125
EXPECTED_VAL_COUNT=125
MAX_TURNS=4
GEPA_CALL_BUDGET=2500
NUM_WORKERS=3
GEPA_SEEDS=(42 2)

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
: "${ELM_API_KEY:?Set ELM_API_KEY for the GPT-4.1 planner.}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY for live Gemini captioning and perception.}"

export PLANNER_API_KEY="${ELM_API_KEY}"
unset DEEPSEEK_API_KEY DEEPSEEK_TOKEN DEEPSEEK_BASE_URL DEEPSEEK_API_BASE PLANNER_API_BASE
export PERCEPTION_MODEL=gemini
export CAPTIONER_MODEL=gemini
export DSPY_AVQA_ALLOWED_TOOLS="ask_caption,ask_perception"
export GEMINI_API_BACKEND="${GEMINI_API_BACKEND:-legacy}"
export GEMINI_RESPONSE_ERROR_SENSITIVE=false

case "${GEMINI_API_BACKEND}" in
  legacy|dspy) ;;
  *)
    echo "GEMINI_API_BACKEND must be either legacy or dspy." >&2
    exit 2
    ;;
esac

for required_file in \
  "${INPUT_JSONL}" \
  "${TRAINSET_JSONL}" \
  "${VALSET_JSONL}" \
  "${CAPTION_CACHE_DIR}/manifest.json" \
  "${PLANNER_CONFIG_YAML}" \
  "${GEMINI_CONFIG_YAML}" \
  "${BASE_PROMPT_YAML}"; do
  if [ ! -f "${required_file}" ]; then
    echo "Required file not found: ${required_file}" >&2
    exit 1
  fi
done

"${PYTHON_BIN}" - "${INPUT_JSONL}" "${EXPECTED_COUNT}" \
  "${TRAINSET_JSONL}" "${EXPECTED_TRAIN_COUNT}" \
  "${VALSET_JSONL}" "${EXPECTED_VAL_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

for path_text, expected_text in zip(sys.argv[1::2], sys.argv[2::2]):
    path = Path(path_text)
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    expected = int(expected_text)
    ids = [str(row.get("id") or "") for row in rows]
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} rows in {path}, found {len(rows)}.")
    if any(not item for item in ids) or len(set(ids)) != expected:
        raise SystemExit(f"Expected {expected} nonempty unique ids in {path}.")
PY

source_run_for_seed() {
  local seed="$1"
  printf '%s_seed%s_calls%s' "${RUN_PREFIX}" "${seed}" "${GEPA_CALL_BUDGET}"
}

output_dir_for_seed() {
  local seed="$1"
  local source_run
  source_run="$(source_run_for_seed "${seed}")"
  printf '%s_best_full_v3' "${source_run}"
}

verify_source_program() {
  local seed="$1"
  local source_run="$2"
  local optimized_prompt_yaml="$3"

  "${PYTHON_BIN}" - \
    "${seed}" \
    "${BASE_PROMPT_YAML}" \
    "${optimized_prompt_yaml}" \
    "${source_run}/compiled_gepa.json" \
    "${source_run}/compiled_gepa_metadata.json" \
    "${source_run}/optimized_planner_optimization_guidance.txt" \
    "${source_run}/planner_optimization_guidance_candidates.jsonl" <<'PY'
import json
import sys
from pathlib import Path

import yaml

(
    seed,
    base_yaml_path,
    optimized_yaml_path,
    compiled_path,
    metadata_path,
    guidance_path,
    candidates_path,
) = sys.argv[1:]

base = yaml.safe_load(Path(base_yaml_path).read_text())
optimized = yaml.safe_load(Path(optimized_yaml_path).read_text())
compiled = json.loads(Path(compiled_path).read_text())
metadata = json.loads(Path(metadata_path).read_text())
guidance_file = Path(guidance_path).read_text().strip()
candidates = [
    json.loads(line)
    for line in Path(candidates_path).read_text().splitlines()
    if line.strip()
]

if metadata.get("optimization_train_only") is not True:
    raise SystemExit(f"Seed {seed}: source run is not marked optimization_train_only.")
if metadata.get("optimize_target") != "planner.optimization_guidance":
    raise SystemExit(f"Seed {seed}: unexpected optimization target in metadata.")

contract = str(base["planner"]["workflow_contract"]).strip()
optimized_contract = str(optimized["planner"]["workflow_contract"]).strip()
compiled_guidance = str(
    compiled["prompt_component_0"]["signature"]["instructions"]
).strip()
optimized_guidance = str(optimized["planner"]["optimization_guidance"]).strip()

if optimized_contract != contract:
    raise SystemExit(f"Seed {seed}: optimized YAML changed the fixed workflow contract.")
if not compiled_guidance or compiled_guidance != optimized_guidance:
    raise SystemExit(f"Seed {seed}: compiled guidance does not match optimized YAML.")
if compiled_guidance != guidance_file:
    raise SystemExit(f"Seed {seed}: compiled guidance does not match optimized guidance text.")
if optimized["planner"]["task_prompt_template"] != base["planner"]["task_prompt_template"]:
    raise SystemExit(f"Seed {seed}: optimized YAML changed the task prompt template.")

template = str(base["planner"]["task_prompt_template"])
contract_token = "{workflow_contract}"
guidance_label = "Optimization guidance:"
guidance_token = "{optimization_guidance}"
if not (
    contract_token in template
    and guidance_label in template
    and guidance_token in template
    and template.index(contract_token) < template.index(guidance_label) < template.index(guidance_token)
):
    raise SystemExit(f"Seed {seed}: task template does not concatenate contract then guidance.")

selected = [row for row in candidates if row.get("whether_selected_as_best") is True]
if len(selected) != 1 or str(selected[0].get("prompt_text", "")).strip() != compiled_guidance:
    raise SystemExit(f"Seed {seed}: compiled guidance is not the uniquely selected best candidate.")

print(
    f"Seed {seed} verified: candidate={selected[0]['candidate_index']}, "
    f"Val125={selected[0]['full_valset_score']}, target=planner.optimization_guidance, "
    "runtime prompt=fixed workflow_contract + optimized guidance"
)
PY
}

build_inference_command() {
  local source_run="$1"
  local output_dir="$2"
  local optimized_prompt_yaml="$3"
  local copied_program="${output_dir}/compiled_gepa.json"
  local -n command_ref="$4"

  command_ref=(
    "${PYTHON_BIN}" DSPy/avqa_dspy_optimize.py
    --algorithm gepa
    --inference-only
    --input-jsonl "${INPUT_JSONL}"
    --trainset-jsonl "${TRAINSET_JSONL}"
    --valset-jsonl "${VALSET_JSONL}"
    --daily-omni-root "${DATA_ROOT}"
    --max-turns "${MAX_TURNS}"
    --planner-config-yaml "${PLANNER_CONFIG_YAML}"
    --gepa-reflection-config-yaml "${PLANNER_CONFIG_YAML}"
    --perception-model gemini
    --perception-config-yaml "${GEMINI_CONFIG_YAML}"
    --captioner-config-yaml "${GEMINI_CONFIG_YAML}"
    --prompt-yaml "${optimized_prompt_yaml}"
    --optimize-target planner.optimization_guidance
    --allowed-tools "${DSPY_AVQA_ALLOWED_TOOLS}"
    --caption-cache-dir "${CAPTION_CACHE_DIR}"
    --caption-cache-scope first_call_only
    --ignore-audio-caption-dir
    --gemini-api-backend "${GEMINI_API_BACKEND}"
    --signature-in-system-prompt
    --caption-placement task
    --output-program "${copied_program}"
    --metadata-json "${output_dir}/inference_only_metadata.json"
    --final-eval-output-jsonl "${output_dir}/output_test.jsonl"
    --final-eval-output-dir "${output_dir}"
    --final-eval-num-threads "${NUM_WORKERS}"
    --final-eval-batch-size "${NUM_WORKERS}"
    --print-config
  )
}

for seed in "${GEPA_SEEDS[@]}"; do
  source_run="$(source_run_for_seed "${seed}")"
  optimized_prompt_yaml="${source_run}/planner_optimization_guidance_GEPA_daily_qa_prompt_v8_free_react_guidance_caption_in_task.yaml"
  for required_source_file in \
    "${source_run}/compiled_gepa.json" \
    "${source_run}/compiled_gepa_metadata.json" \
    "${source_run}/optimized_planner_optimization_guidance.txt" \
    "${source_run}/planner_optimization_guidance_candidates.jsonl" \
    "${optimized_prompt_yaml}"; do
    if [ ! -f "${required_source_file}" ]; then
      echo "Required source artifact not found: ${required_source_file}" >&2
      exit 1
    fi
  done
  verify_source_program "${seed}" "${source_run}" "${optimized_prompt_yaml}"
done

echo "Full-v3 inference seeds: ${GEPA_SEEDS[*]}"
echo "Parallel jobs: ${#GEPA_SEEDS[@]}; workers per job: ${NUM_WORKERS}"
echo "Planner prompt: fixed workflow_contract followed by each seed's optimized guidance"
echo "Caption/perception: first caption exact-cache, later calls live Gemini"

if [ "${DRY_RUN:-false}" = true ]; then
  for seed in "${GEPA_SEEDS[@]}"; do
    source_run="$(source_run_for_seed "${seed}")"
    output_dir="$(output_dir_for_seed "${seed}")"
    optimized_prompt_yaml="${source_run}/planner_optimization_guidance_GEPA_daily_qa_prompt_v8_free_react_guidance_caption_in_task.yaml"
    build_inference_command "${source_run}" "${output_dir}" "${optimized_prompt_yaml}" inference_cmd
    printf 'Seed %s output directory: %s\nCommand:' "${seed}" "${output_dir}"
    printf ' %q' "${inference_cmd[@]}"
    printf '\n'
  done
  exit 0
fi

pids=()
output_dirs=()
cleanup_children() {
  for pid in "${pids[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap cleanup_children EXIT INT TERM

for seed in "${GEPA_SEEDS[@]}"; do
  source_run="$(source_run_for_seed "${seed}")"
  output_dir="$(output_dir_for_seed "${seed}")"
  optimized_prompt_yaml="${source_run}/planner_optimization_guidance_GEPA_daily_qa_prompt_v8_free_react_guidance_caption_in_task.yaml"
  copied_program="${output_dir}/compiled_gepa.json"
  mkdir -p "${output_dir}"
  if [ -f "${copied_program}" ]; then
    if ! cmp -s "${source_run}/compiled_gepa.json" "${copied_program}"; then
      echo "Refusing to resume seed ${seed}: ${copied_program} differs from the source program." >&2
      exit 2
    fi
  else
    cp "${source_run}/compiled_gepa.json" "${copied_program}"
  fi
  build_inference_command "${source_run}" "${output_dir}" "${optimized_prompt_yaml}" inference_cmd
  (
    echo "Seed ${seed} inference started: $(date -Is)"
    echo "Source program: ${source_run}/compiled_gepa.json"
    echo "Prompt YAML: ${optimized_prompt_yaml}"
    echo "Output directory: ${output_dir}"
    echo "Workers: ${NUM_WORKERS}"
    "${inference_cmd[@]}"
    echo "Seed ${seed} inference finished: $(date -Is)"
  ) > >(tee -a "${output_dir}/run.log") 2>&1 &
  pids+=("$!")
  output_dirs+=("${output_dir}")
done

failures=0
for index in "${!pids[@]}"; do
  if wait "${pids[index]}"; then
    echo "Completed: ${output_dirs[index]}"
  else
    echo "Failed: ${output_dirs[index]} (see run.log)" >&2
    failures=$((failures + 1))
  fi
done
trap - EXIT INT TERM

if [ "${failures}" -ne 0 ]; then
  echo "${failures} of ${#GEPA_SEEDS[@]} full-v3 inference jobs failed." >&2
  exit 1
fi
echo "Both guidance-only GEPA full-v3 inference jobs completed."
