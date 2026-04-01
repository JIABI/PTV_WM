#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/si_d.yaml
note_outputs
echo "[paper] expected artifacts: outputs/figures/FigureS4_* FigureS5_*"
echo "[paper] expected artifacts: outputs/tables/TableS6_*"
