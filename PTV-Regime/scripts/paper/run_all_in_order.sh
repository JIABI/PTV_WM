#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"${SCRIPT_DIR}/00_smoke.sh"
"${SCRIPT_DIR}/run_main_in_order.sh"
"${SCRIPT_DIR}/run_si_in_order.sh"
