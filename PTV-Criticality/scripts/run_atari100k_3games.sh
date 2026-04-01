#!/usr/bin/env bash
set -euo pipefail

# Run Atari-100k style experiments (baseline pipelines).
# Note: requires `pip install gymnasium[atari] ale-py`.

GAMES=(Pong Breakout Qbert)

for G in "${GAMES[@]}"; do
  echo "=== Running Atari100k game: ${G} ==="
  # DrQv2 baseline by default.
  python -m fatewm.experiments.atari100k.train env=atari100k algo=drqv2 method=uniform \
    device=cuda env.game=${G} \
    train.total_steps=100000 replay.warmup_steps=2000 eval.interval=10000 eval.episodes=10 \
    eval.policy=algo
done
