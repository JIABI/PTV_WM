#!/usr/bin/env bash
set -euo pipefail

# Run Scheme-1 (oracle-based) on three Procgen environments.
# Note: requires `pip install procgen`.

ENVS=(coinrun jumper fruitbot)

for E in "${ENVS[@]}"; do
  echo "=== Running Procgen env: ${E} ==="
  python -m fatewm.experiments.procgen.train env=procgen_coinrun method=scheme1_procgen algo=minimal \
    device=cuda env.env_name=${E} \
    env.env_name=coinrun env.num_levels=1 env.start_level=0 env.episode_len=1000 \
  interface.hysteresis.margin=0.0 \
  +interface.eps=0.1 +interface.act_temperature=1.0 \
  train.total_steps=200000 eval.interval=10000 eval.episodes=10 eval.policy=interface
done
