#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SHOW_WRONG="${SHOW_WRONG:-5}"

usage() {
  cat >&2 <<EOF
Usage:
  $0 <directory-or-glob> [<directory-or-glob> ...]

Examples:
  $0 '/mnt/ceph_rbd/data/avqa_project/daily_omni/daily_omni_dspy_qwen_v6MULTITURN_gepa_qwen_seed_1234_deepseek_seed_7_gepa_seed*'
  SHOW_WRONG=0 $0 /path/to/run_seed*

Each matched directory is evaluated using its output_test.jsonl.
EOF
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

shopt -s nullglob

input_files=()
for pattern in "$@"; do
  matches=()
  for match in ${pattern}; do
    matches+=("${match}")
  done

  if [ "${#matches[@]}" -eq 0 ]; then
    echo "No directory matched pattern: ${pattern}" >&2
    continue
  fi

  for match in "${matches[@]}"; do
    if [ -d "${match}" ]; then
      input_file="${match%/}/output_test.jsonl"
    else
      input_file="${match}"
    fi

    if [ -f "${input_file}" ]; then
      input_files+=("${input_file}")
    else
      echo "Missing output_test.jsonl: ${input_file}" >&2
    fi
  done
done

if [ "${#input_files[@]}" -eq 0 ]; then
  echo "No output_test.jsonl files found." >&2
  exit 1
fi

IFS=$'\n' input_files=($(printf '%s\n' "${input_files[@]}" | sort -u))
unset IFS

for input_file in "${input_files[@]}"; do
  "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/react_eval.py" \
    --input_file "${input_file}" \
    --show_wrong "${SHOW_WRONG}"
done
