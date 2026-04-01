#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
"${SCRIPT_DIR}/10_si_a_datasets.sh"
"${SCRIPT_DIR}/11_si_b_baselines.sh"
"${SCRIPT_DIR}/12_si_c_lic_boids_extended.sh"
"${SCRIPT_DIR}/13_si_d_counterfactual.sh"
"${SCRIPT_DIR}/14_si_e_cross_domain_extended.sh"
"${SCRIPT_DIR}/15_si_f_uav_extended.sh"
"${SCRIPT_DIR}/16_si_g_capacity_residual.sh"
"${SCRIPT_DIR}/17_si_h_antisteg_leakage.sh"
"${SCRIPT_DIR}/18_si_i_event_ood.sh"
