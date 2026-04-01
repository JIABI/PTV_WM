#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/si_f.yaml
note_outputs
echo "[paper] expected artifacts: outputs/figures/FigureS8_* FigureS9_*"
echo "[paper] expected artifacts: outputs/tables/TableS8_* TableS9_*"
