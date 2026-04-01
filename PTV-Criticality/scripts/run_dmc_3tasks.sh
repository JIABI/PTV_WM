#!/usr/bin/env bash
set -euo pipefail

# Run Scheme-1 on three standard DMC tasks.
# Edit algo/method knobs as needed.

TASKS=(cartpole_swingup cheetah_run walker_walk)

for T in "${TASKS[@]}"; do
  echo "=== Running DMC task: ${T} ==="
  python -m fatewm.experiments.dmc.train env=dmc method=scheme1_dmc algo=minimal \
    device=cuda env.task=${T} \
    method.oracle_horizon=5 method.candidate_sigma=0.5 \
    interface.refine.enabled=true interface.refine.method=mirror \
    interface.refine.eta=1.0 interface.refine.temperature=2.0 \
    interface.hysteresis.margin=0.0 \
    eval.policy=interface
done
