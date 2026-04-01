#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/si_c.yaml
note_outputs
echo "[paper] expected artifacts: outputs/figures/FigureS1_* FigureS2_* FigureS3_*"
echo "[paper] expected artifacts: outputs/tables/TableS5_*"
