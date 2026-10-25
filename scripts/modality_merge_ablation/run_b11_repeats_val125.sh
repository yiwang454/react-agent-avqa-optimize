#!/usr/bin/env bash
set -euo pipefail

# B11 is B6 with only the o3 reasoning effort changed from high to medium.
# Run repeat1-3 concurrently, with three question workers in each repeat.
# Reusing the same output directories resumes per-question checkpoints.

if [ "$#" -ne 0 ]; then
  echo "Usage: bash $0 (optionally set OUTPUT_BASE)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/ceph_rbd/data/avqa_project/daily_omni/modality_merge_ablation_b11_selectedVal125_repeats}"
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

trap 'cleanup_children; exit 130' INT
trap 'cleanup_children; exit 143' TERM
trap 'cleanup_children' EXIT

echo "Starting B11 repeat1-3 in parallel"
echo "Output root: ${OUTPUT_BASE}"
echo "Per-repeat concurrency: threads=3, batch=3"

for repeat_name in "${REPEATS[@]}"; do
  output_dir="${OUTPUT_BASE}/${repeat_name}/b11"
  launcher_log="${output_dir}/launcher.log"
  mkdir -p "${output_dir}"
  echo "Launching ${repeat_name}/B11; log=${launcher_log}"
  setsid --wait env \
    OUTPUT_DIR_OVERRIDE="${output_dir}" \
    INFERENCE_NUM_THREADS=3 \
    INFERENCE_BATCH_SIZE=3 \
    "${SCRIPT_DIR}/run_b11_val125.sh" > "${launcher_log}" 2>&1 &
  child_pids+=("$!")
  child_ids+=("${repeat_name}")
done

failed=0
for index in "${!child_pids[@]}"; do
  repeat_name="${child_ids[index]}"
  launcher_log="${OUTPUT_BASE}/${repeat_name}/b11/launcher.log"
  if wait "${child_pids[index]}"; then
    echo "Completed ${repeat_name}/B11"
  else
    status=$?
    failed=1
    echo "FAILED ${repeat_name}/B11 (exit ${status}); log=${launcher_log}" >&2
    tail -n 40 "${launcher_log}" >&2 || true
  fi
  child_pids[index]=""
done

trap - INT TERM EXIT
if [ "${failed}" -ne 0 ]; then
  exit 1
fi
echo "Finished B11 repeat1-3."
