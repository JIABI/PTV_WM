#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/si_g.yaml
note_outputs
echo "[paper] expected artifacts: outputs/figures/FigureS10_* FigureS11_* FigureS12_*"
echo "[paper] expected artifacts: outputs/tables/TableS10_*"
