#!/usr/bin/env bash
set -euo pipefail

# Rerun only the B11 repeat-question records that previously used >6 tools.
# Each rerun keeps B11's configuration except for a six-valid-call budget.
# The original 30-call results/checkpoints are read-only and never reused.

if [ "$#" -ne 0 ]; then
  echo "Usage: bash $0 (optionally set SOURCE_BASE or OUTPUT_BASE)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_BASE="${SOURCE_BASE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_b11_selectedVal125_repeats}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_b11_over6_budget6_reruns}"
REPEATS=(repeat1 repeat2 repeat3)
child_pids=()
child_ids=()

cleanup_children() {
  local pid attempt any_active
  local active_pids=()
  trap - INT TERM EXIT
  for pid in "${child_pids[@]}"; do
    if [ -n "${pid}" ]; then
      active_pids+=("${pid}")
    fi
  done
  if [ "${#active_pids[@]}" -eq 0 ]; then
    return
  fi
  for pid in "${active_pids[@]}"; do
    kill -TERM -- "-${pid}" 2>/dev/null || true
  done
  for attempt in {1..20}; do
    any_active=0
    for pid in "${active_pids[@]}"; do
      if kill -0 -- "-${pid}" 2>/dev/null; then
        any_active=1
        break
      fi
    done
    if [ "${any_active}" -eq 0 ]; then
      break
    fi
    sleep 0.25
  done
  for pid in "${active_pids[@]}"; do
    if kill -0 -- "-${pid}" 2>/dev/null; then
      kill -KILL -- "-${pid}" 2>/dev/null || true
    fi
  done
  wait "${active_pids[@]}" 2>/dev/null || true
}

# Verify the source set before writing anything, so this script cannot silently
# rerun a different selection if the original B11 outputs have changed.
for repeat_name in "${REPEATS[@]}"; do
  source_jsonl="${SOURCE_BASE}/${repeat_name}/b11/output_test.jsonl"
  if [ ! -f "${source_jsonl}" ]; then
    echo "Missing original B11 output: ${source_jsonl}" >&2
    exit 2
  fi
  if [ "$(wc -l < "${source_jsonl}")" -ne 125 ]; then
    echo "Expected 125 rows in ${source_jsonl}" >&2
    exit 2
  fi
  case "${repeat_name}" in
    repeat1) expected_ids="37UUg0fwzsw-1,TwsSrD8g3HA-2" ;;
    repeat2) expected_ids="AYW_zgrLc6U-1,TwsSrD8g3HA-2" ;;
    repeat3) expected_ids="TwsSrD8g3HA-2" ;;
  esac
  actual_ids="$(jq -r 'select(.question_data.metrics.tool_calls.budgeted_total > 6) | .question_data.question_id' "${source_jsonl}" | LC_ALL=C sort | paste -sd, -)"
  if [ "${actual_ids}" != "${expected_ids}" ]; then
    echo "Over-six IDs changed in ${source_jsonl}: expected ${expected_ids}, got ${actual_ids}" >&2
    exit 2
  fi
done

trap 'cleanup_children; exit 130' INT
trap 'cleanup_children; exit 143' TERM
trap 'cleanup_children' EXIT

echo "Rerunning five B11 repeat-question records (three unique IDs) with max_turns=6"
echo "Output root: ${OUTPUT_BASE}"
echo "Repeat concurrency: 3 parallel repeats; per-repeat threads=3, batch=3"
for repeat_name in "${REPEATS[@]}"; do
  case "${repeat_name}" in
    repeat1) sample_ids="37UUg0fwzsw-1,TwsSrD8g3HA-2" ;;
    repeat2) sample_ids="AYW_zgrLc6U-1,TwsSrD8g3HA-2" ;;
    repeat3) sample_ids="TwsSrD8g3HA-2" ;;
  esac
  output_dir="${OUTPUT_BASE}/${repeat_name}/b11-tail6"
  launcher_log="${output_dir}/launcher.log"
  mkdir -p "${output_dir}"
  echo "Launching ${repeat_name}: ${sample_ids}; log=${launcher_log}"
  setsid --wait env \
    SAMPLE_IDS="${sample_ids}" \
    OUTPUT_DIR_OVERRIDE="${output_dir}" \
    INFERENCE_NUM_THREADS=3 \
    INFERENCE_BATCH_SIZE=3 \
    "${SCRIPT_DIR}/run_val125_common.sh" B11-TAIL6 > "${launcher_log}" 2>&1 &
  child_pids+=("$!")
  child_ids+=("${repeat_name}")
done

failed=0
for index in "${!child_pids[@]}"; do
  repeat_name="${child_ids[index]}"
  launcher_log="${OUTPUT_BASE}/${repeat_name}/b11-tail6/launcher.log"
  if wait "${child_pids[index]}"; then
    echo "Completed ${repeat_name}"
  else
    status=$?
    failed=1
    echo "FAILED ${repeat_name} (exit ${status}); log=${launcher_log}" >&2
    tail -n 40 "${launcher_log}" >&2 || true
  fi
  child_pids[index]=""
done

trap - INT TERM EXIT
if [ "${failed}" -ne 0 ]; then
  exit 1
fi
echo "Finished all B11 over-six reruns."
