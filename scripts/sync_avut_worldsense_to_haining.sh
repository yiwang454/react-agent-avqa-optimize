#!/usr/bin/env bash
# Temporarily sync the AVUT direct-input directory and WorldSense to haining.
# The two plain rsync processes run in parallel.  This script deliberately does
# not use SSH connection multiplexing or --delete.

set -euo pipefail

readonly REMOTE_HOST="haining"
readonly REMOTE_ROOT="/mnt/ceph_rbd/data/avqa_project"
readonly AVUT_SOURCE="/mnt/ceph_rbd/data/avqa_project/avut/mllm_instruct_opt/avut_gemini2.5flash_direct_safe_duration_sorted_input"
readonly WORLDSENSE_SOURCE="/mnt/ceph_rbd/data/avqa_project/WorldSense"

[[ -d "${AVUT_SOURCE}" ]] || { echo "Missing source: ${AVUT_SOURCE}" >&2; exit 1; }
[[ -d "${WORLDSENSE_SOURCE}" ]] || { echo "Missing source: ${WORLDSENSE_SOURCE}" >&2; exit 1; }

# Each transfer opens a normal SSH session and may prompt for the password.
# The remote rsync path creates the only missing parent directory as part of
# the AVUT transfer, without an additional ssh invocation.
rsync -aH --partial --append-verify --info=progress2 \
  --rsync-path="mkdir -p '${REMOTE_ROOT}/avut/mllm_instruct_opt' && rsync" \
  "${AVUT_SOURCE}" "${REMOTE_HOST}:${REMOTE_ROOT}/avut/mllm_instruct_opt/" &
AVUT_PID=$!

rsync -aH --partial --append-verify --info=progress2 \
  "${WORLDSENSE_SOURCE}" "${REMOTE_HOST}:${REMOTE_ROOT}/" &
WORLDSENSE_PID=$!

terminate_children() {
  kill "${AVUT_PID}" "${WORLDSENSE_PID}" 2>/dev/null || true
  wait "${AVUT_PID}" 2>/dev/null || true
  wait "${WORLDSENSE_PID}" 2>/dev/null || true
}
trap 'terminate_children; exit 130' INT TERM HUP

status=0
wait "${AVUT_PID}" || status=1
wait "${WORLDSENSE_PID}" || status=1
exit "${status}"
