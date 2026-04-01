#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/si_i.yaml
note_outputs
echo "[paper] expected artifacts: outputs/figures/FigureS16_* FigureS17_* FigureS18_*"
echo "[paper] expected artifacts: outputs/tables/TableS12_*"
