#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/si_b.yaml
note_outputs
echo "[paper] expected artifacts: outputs/tables/TableS3_* outputs/tables/TableS4_*"
