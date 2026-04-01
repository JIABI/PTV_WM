#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/common.sh"
run_section configs/sweep/main_section5.yaml
note_outputs
echo "[paper] expected artifacts: outputs/tables/Table1_benchmark_protocol_metrics.*"
