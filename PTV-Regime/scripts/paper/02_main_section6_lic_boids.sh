#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/main_section6.yaml
note_outputs
echo "[paper] expected artifacts: outputs/tables/Table2_lic_boids_quantitative_summary.*"
echo "[paper] expected artifacts: outputs/figures/Figure2_*"
