#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/main_section7.yaml
note_outputs
echo "[paper] expected artifacts: outputs/tables/Table3_cross_domain_predictive_stability.*"
echo "[paper] expected artifacts: outputs/figures/Figure3_*"
