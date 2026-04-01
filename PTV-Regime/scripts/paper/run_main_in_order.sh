#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"${SCRIPT_DIR}/01_main_section5_setup.sh"
"${SCRIPT_DIR}/02_main_section6_lic_boids.sh"
"${SCRIPT_DIR}/03_main_section7_cross_domain.sh"
"${SCRIPT_DIR}/04_main_section8_lic_uav.sh"
"${SCRIPT_DIR}/05_main_section9_defense.sh"
