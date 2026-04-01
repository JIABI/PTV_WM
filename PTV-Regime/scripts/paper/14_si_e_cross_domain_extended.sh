#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/si_e.yaml
note_outputs
echo "[paper] expected artifacts: outputs/figures/FigureS6_* FigureS7_*"
echo "[paper] expected artifacts: outputs/tables/TableS7_*"
