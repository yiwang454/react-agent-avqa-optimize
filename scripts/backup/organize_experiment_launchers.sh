#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/backup/organize_experiment_launchers.sh [--dry-run|--apply]

Without arguments (or with --dry-run), print the planned COPY/MOVE/KEEP
operations without changing any launcher. Pass --apply to create the category
directories, copy the four primary launchers and two shared wrappers, and move
the archived launchers.
EOF
}

MODE="dry-run"
case "${1:-}" in
  ""|--dry-run)
    ;;
  --apply)
    MODE="apply"
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

if (( $# > 1 )); then
  usage >&2
  exit 2
fi

BACKUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPTS_DIR="$(cd "${BACKUP_DIR}/.." && pwd -P)"

if [[ "$(basename "${BACKUP_DIR}")" != "backup" || "$(basename "${SCRIPTS_DIR}")" != "scripts" ]]; then
  echo "ERROR: this organizer must remain directly under scripts/backup/." >&2
  exit 1
fi

readonly CATEGORY_GPT_G0="01_gemini_gpt_none_gepa_and_g0"
readonly CATEGORY_GPT_G1_G3="02_gemini_gpt_g1_g2_g3"
readonly CATEGORY_GPT_QWEN_GEMINI="03a_gpt_qwen_gemini_mixed"
readonly CATEGORY_DEEPSEEK_QWEN_V8="03b_deepseek_qwen_v8_g0"
readonly CATEGORY_LEGACY_QWEN_V5_GEPA="03c_legacy_qwen_v5_gepa"
readonly CATEGORY_TEMPLATES_V5="04a_v5_template_baselines"
readonly CATEGORY_TEMPLATES_V8="04b_v8_template_baselines"

# These four primary launchers remain in scripts/ and are copied to backup/.
copy_gpt_g0=(
  "run_dspy_react_agent_daily_qwen_v8_GPT4.1_gemini_captionInTask.sh"
  "run_dspy_react_agent_daily_qwen_v8_GPT_gemini_captionInTask.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_planner.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt5.4_v8_gemini_caption_in_task_planner.sh"
)

# These shared wrappers are archived with the related GPT/Gemini G0 family,
# but are also retained in scripts/ because the four primary launchers need
# them at their original paths.
copy_gpt_g0_dependencies=(
  "run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_common_0721.sh"
)

move_gpt_g0=(
  "run_dspy_react_agent_avut_gepa_elm_gpt_v8_gemini_caption_in_task_planner.sh"
  "run_dspy_react_agent_daily_complete_unanswered_gpt4_1.sh"
  "run_dspy_react_agent_daily_complete_unanswered_gpt5_4.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt4.1_v8_gemini_caption_in_task_captioner_common_0721.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt4.1_v8_gemini_caption_in_task_planner_captioner_common_0721.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_captioner.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_captioner_0721.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_captioner_privileged_0721.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_planner_captioner.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_planner_captioner_0721.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt_v8_gemini_caption_in_task_planner_event_sequence.sh"
  "run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_g0_gpt4_1_planner.sh"
  "run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_g0_gpt5_4_planner.sh"
)

move_gpt_g1_g3=(
  "run_dspy_react_agent_daily_gepa_densified_gpt4_1_planner.sh"
  "run_dspy_react_agent_daily_gepa_densified_gpt4_1_planner_captioner.sh"
  "run_dspy_react_agent_daily_gepa_densified_gpt4_1_planner_repeat.sh"
  "run_dspy_react_agent_daily_gepa_densified_gpt5_4_planner.sh"
  "run_dspy_react_agent_daily_gepa_densified_gpt5_4_planner_captioner.sh"
  "run_dspy_react_agent_daily_gepa_elm_gpt4_1_v8_gemini_caption_in_task_captioner_g1.sh"
  "run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_g3_gpt4_1_planner.sh"
  "run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_g3_gpt5_4_planner.sh"
  "run_dspy_react_agent_daily_inference_g3_gpt5_4_planner_best_candidate.sh"
)

move_gpt_qwen_gemini=(
  "run_dspy_react_agent_daily_elm_gpt4_1_qwen_cached_gemini_v8_captionInTask_val125.sh"
  "run_dspy_react_agent_daily_gepa_g0_elm_gpt4.1_qwen_gemini_v8_caption_in_task_planner_gpt5_4_reflection.sh"
  "run_dspy_react_agent_daily_gepa_g0_elm_gpt4.1_qwen_gemini_v8_free_caption_in_task_planner_gpt5_4_reflection.sh"
  "run_dspy_react_agent_daily_inference_g0_elm_gpt4_1_qwen_cached_gemini_v8_caption_in_task_planner_gpt5_4_reflection.sh"
)

move_deepseek_qwen_v8=(
  "run_dspy_react_agent_daily_gepa_g0_deepseek_qwen_cached_gemini_v8_caption_in_task_planner_deepseek_reflection_thinking_enable.sh"
  "run_dspy_react_agent_daily_gepa_g0_deepseek_qwen_cached_gemini_v8_caption_in_task_planner_gpt5_4_reflection.sh"
  "run_dspy_react_agent_daily_gepa_g0_deepseek_qwen_gemini_v8_free_caption_in_task_planner_gpt5_4_reflection.sh"
  "run_dspy_react_agent_daily_gepa_g0p_deepseek_qwen_v8_fixed_caption_in_task_planner_gpt5_4_reflection.sh"
  "run_dspy_react_agent_daily_gepa_g0p_deepseek_qwen_v8_free_caption_in_task_planner_gpt5_4_reflection.sh"
  "run_dspy_react_agent_daily_inference_g0_deepseek_qwen_cached_gemini_v8_caption_in_task_planner_reflectionDeepseek.sh"
)

move_legacy_qwen_v5_gepa=(
  "run_dspy_react_agent_daily_gepa_deepseek_qwen_v5_gemini_caption_temp0p7.sh"
  "run_dspy_react_agent_daily_gepa_qwen2.sh"
  "run_dspy_react_agent_daily_gepa_seed18_repeat2_qwen2.sh"
  "run_dspy_react_agent_daily_gepa_temp0p7_qwen_v5_gemini_caption.sh"
)

move_templates_v5=(
  "run_dspy_react_agent_daily_qwen_v5_gemini_caption.sh"
  "run_dspy_react_agent_daily_qwen_v5_qwen_seed.sh"
)

move_templates_v8=(
  "run_dspy_react_agent_daily_qwen_v8_cached_gemini_captionInTask_seed.sh"
  "run_dspy_react_agent_daily_qwen_v8_gemini_captionInTask_seed.sh"
  "run_dspy_react_agent_daily_qwen_v8_gemini_caption_seed.sh"
  "run_dspy_react_agent_daily_qwen_v8_qwen_seed.sh"
  "run_dspy_react_agent_daily_qwen_v8free_gemini_captionInTask_seed.sh"
  "run_dspy_react_agent_daily_qwen_v8free_gemini_caption_seed.sh"
  "run_dspy_react_agent_daily_qwen_v8free_qwen_seed.sh"
  "run_dspy_react_agent_daily_qwen_v8simplified_gemini_caption_seed.sh"
)

# These dispatch wrappers remain at scripts/ because they form the dependency
# chain used by archived G1/G3 families and by the copied common_0721 wrapper.
keep_in_scripts=(
  "run_dspy_react_agent_daily_gepa_densified_common.sh"
  "run_dspy_react_agent_daily_gepa_v8free_gemini_captionInTask_common.sh"
)

declare -A seen=()
declare -A action_for=()
declare -A category_for=()

register_files() {
  local action="$1"
  local category="$2"
  shift 2
  local filename
  for filename in "$@"; do
    if [[ -n "${seen[${filename}]:-}" ]]; then
      echo "ERROR: duplicate manifest entry: ${filename}" >&2
      exit 1
    fi
    seen["${filename}"]=1
    action_for["${filename}"]="${action}"
    category_for["${filename}"]="${category}"
  done
}

register_files "COPY" "${CATEGORY_GPT_G0}" "${copy_gpt_g0[@]}"
register_files "COPY" "${CATEGORY_GPT_G0}" "${copy_gpt_g0_dependencies[@]}"
register_files "MOVE" "${CATEGORY_GPT_G0}" "${move_gpt_g0[@]}"
register_files "MOVE" "${CATEGORY_GPT_G1_G3}" "${move_gpt_g1_g3[@]}"
register_files "MOVE" "${CATEGORY_GPT_QWEN_GEMINI}" "${move_gpt_qwen_gemini[@]}"
register_files "MOVE" "${CATEGORY_DEEPSEEK_QWEN_V8}" "${move_deepseek_qwen_v8[@]}"
register_files "MOVE" "${CATEGORY_LEGACY_QWEN_V5_GEPA}" "${move_legacy_qwen_v5_gepa[@]}"
register_files "MOVE" "${CATEGORY_TEMPLATES_V5}" "${move_templates_v5[@]}"
register_files "MOVE" "${CATEGORY_TEMPLATES_V8}" "${move_templates_v8[@]}"
register_files "KEEP" "scripts" "${keep_in_scripts[@]}"

# Audit the current top-level experiment-runner namespace. This glob is used
# only for validation; movement always follows the explicit manifests above.
shopt -s nullglob
for path in "${SCRIPTS_DIR}"/run_dspy_react_agent*.sh; do
  filename="$(basename "${path}")"
  if [[ -z "${seen[${filename}]:-}" ]]; then
    echo "ERROR: unclassified top-level experiment script: ${filename}" >&2
    exit 1
  fi
done
shopt -u nullglob

categories=(
  "${CATEGORY_GPT_G0}"
  "${CATEGORY_GPT_G1_G3}"
  "${CATEGORY_GPT_QWEN_GEMINI}"
  "${CATEGORY_DEEPSEEK_QWEN_V8}"
  "${CATEGORY_LEGACY_QWEN_V5_GEPA}"
  "${CATEGORY_TEMPLATES_V5}"
  "${CATEGORY_TEMPLATES_V8}"
)

for category in "${categories[@]}"; do
  destination_dir="${BACKUP_DIR}/${category}"
  if [[ -e "${destination_dir}" && ! -d "${destination_dir}" ]]; then
    echo "ERROR: category path exists but is not a directory: ${destination_dir}" >&2
    exit 1
  fi
done

preflight_file() {
  local filename="$1"
  local action="${action_for[${filename}]}"
  local category="${category_for[${filename}]}"
  local source_path="${SCRIPTS_DIR}/${filename}"

  if [[ "${action}" == "KEEP" ]]; then
    if [[ ! -f "${source_path}" ]]; then
      echo "ERROR: required file to keep is missing: ${source_path}" >&2
      exit 1
    fi
    return
  fi

  local destination_path="${BACKUP_DIR}/${category}/${filename}"
  if [[ "${action}" == "COPY" ]]; then
    if [[ ! -f "${source_path}" ]]; then
      echo "ERROR: primary launcher is missing: ${source_path}" >&2
      exit 1
    fi
    if [[ -e "${destination_path}" && ! -f "${destination_path}" ]]; then
      echo "ERROR: destination exists but is not a regular file: ${destination_path}" >&2
      exit 1
    fi
    if [[ -f "${destination_path}" ]] && ! cmp -s -- "${source_path}" "${destination_path}"; then
      echo "ERROR: archived copy differs from source: ${destination_path}" >&2
      exit 1
    fi
    return
  fi

  if [[ -f "${source_path}" && -e "${destination_path}" ]]; then
    if [[ -f "${destination_path}" ]] && cmp -s -- "${source_path}" "${destination_path}"; then
      echo "ERROR: move source and destination both exist: ${filename}" >&2
    else
      echo "ERROR: move destination already exists with different content or type: ${destination_path}" >&2
    fi
    exit 1
  fi
  if [[ ! -e "${source_path}" && ! -f "${destination_path}" ]]; then
    echo "ERROR: move source is missing and no archived file exists: ${filename}" >&2
    exit 1
  fi
  if [[ -e "${source_path}" && ! -f "${source_path}" ]]; then
    echo "ERROR: move source is not a regular file: ${source_path}" >&2
    exit 1
  fi
}

for filename in "${!seen[@]}"; do
  preflight_file "${filename}"
done

print_or_apply() {
  local filename="$1"
  local action="${action_for[${filename}]}"
  local category="${category_for[${filename}]}"
  local source_path="${SCRIPTS_DIR}/${filename}"

  if [[ "${action}" == "KEEP" ]]; then
    printf 'KEEP  %s\n' "${source_path}"
    return
  fi

  local destination_path="${BACKUP_DIR}/${category}/${filename}"
  if [[ "${action}" == "COPY" && -f "${destination_path}" ]]; then
    printf 'KEEP  %s (identical archived copy)\n' "${destination_path}"
    return
  fi
  if [[ "${action}" == "MOVE" && ! -e "${source_path}" && -f "${destination_path}" ]]; then
    printf 'KEEP  %s (already archived)\n' "${destination_path}"
    return
  fi

  printf '%-5s %s -> %s\n' "${action}" "${source_path}" "${destination_path}"
  if [[ "${MODE}" == "apply" ]]; then
    if [[ "${action}" == "COPY" ]]; then
      cp -p -- "${source_path}" "${destination_path}"
      if ! cmp -s -- "${source_path}" "${destination_path}"; then
        echo "ERROR: copied file failed content verification: ${destination_path}" >&2
        exit 1
      fi
    else
      mv -- "${source_path}" "${destination_path}"
    fi
  fi
}

if [[ "${MODE}" == "apply" ]]; then
  for category in "${categories[@]}"; do
    mkdir -p -- "${BACKUP_DIR}/${category}"
  done
else
  echo "DRY RUN: no files or directories will be changed."
fi

# Fixed ordering keeps the preview readable and reproducible.
for filename in "${copy_gpt_g0[@]}"; do print_or_apply "${filename}"; done
for filename in "${copy_gpt_g0_dependencies[@]}"; do print_or_apply "${filename}"; done
for filename in "${move_gpt_g0[@]}"; do print_or_apply "${filename}"; done
for filename in "${move_gpt_g1_g3[@]}"; do print_or_apply "${filename}"; done
for filename in "${move_gpt_qwen_gemini[@]}"; do print_or_apply "${filename}"; done
for filename in "${move_deepseek_qwen_v8[@]}"; do print_or_apply "${filename}"; done
for filename in "${move_legacy_qwen_v5_gepa[@]}"; do print_or_apply "${filename}"; done
for filename in "${move_templates_v5[@]}"; do print_or_apply "${filename}"; done
for filename in "${move_templates_v8[@]}"; do print_or_apply "${filename}"; done
for filename in "${keep_in_scripts[@]}"; do print_or_apply "${filename}"; done

if [[ "${MODE}" == "apply" ]]; then
  echo "Done. The four primary launchers remain in scripts/; all listed archive operations completed."
else
  echo "Dry run complete. Re-run with --apply to perform these operations."
fi
