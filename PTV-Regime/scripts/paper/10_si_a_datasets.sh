#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/si_a.yaml
note_outputs
echo "[paper] expected artifacts: outputs/tables/TableS1_* outputs/tables/TableS2_*"
