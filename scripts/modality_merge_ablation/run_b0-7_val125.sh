#!/usr/bin/env bash
set -euo pipefail

# Compatibility name from the first Stage B suite. Active v2 runs B0-B6 + B5-flex.
exec "$(dirname "$0")/run_stage_b_v2_val125.sh" "$@"
