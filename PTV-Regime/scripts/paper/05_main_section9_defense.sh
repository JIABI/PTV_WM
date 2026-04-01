#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/main_section9.yaml
note_outputs
echo "[paper] expected artifacts: outputs/tables/Table5_ablation_and_stress_test_summary.*"
echo "[paper] expected artifacts: outputs/figures/Figure5_*"
