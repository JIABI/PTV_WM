#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/main_section8.yaml
note_outputs
echo "[paper] expected artifacts: outputs/tables/Table4_lic_uav_control_summary.*"
echo "[paper] expected artifacts: outputs/figures/Figure4_*"
