#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/si_h.yaml
note_outputs
echo "[paper] expected artifacts: outputs/figures/FigureS13_* FigureS14_* FigureS15_*"
echo "[paper] expected artifacts: outputs/tables/TableS11_*"
