# RRRM-WM

Reference implementation of **Resource-Rational Risk Minimization (RRRM)** for closed-loop reliable world models.

This repository is aligned to the paper's core design: a fixed decision interface, a differentiable decision-criticality mediator, horizon-wise tail-risk estimation, budgeted selective strong correction, and optional teacher/oracle supervision on the same candidate pool. The implementation follows the paper's main method block and evaluation protocol rather than the earlier placeholder app scaffold.

## What is implemented

- A **self-contained RRRM training stack** with:
  - multi-horizon latent rollouts
  - margin/listwise decision-criticality mediators
  - CVaR-style horizon risk estimation
  - budgeted horizon allocation with a trainable router
  - four-way fate prediction for diagnostics
  - teacher EMA support for benchmark-style reference energies
- A **toy diagnostic environment** for smoke tests and mechanism checks.
- Optional wrappers for **DMC**, **Procgen**, and **Atari-100k** style environments.
- A unified evaluation path reporting rollout return, tail metrics, drift, switch rate, and ABA / ping-pong behavior.

## Repository layout

```text
src/fatewm/
  algos/        lightweight world-model adapters
  core/         RRRM losses, router, allocation, metrics, interfaces
  envs/         toy + optional DMC / Procgen / Atari wrappers
  experiments/  Hydra entrypoints for toy, DMC, Procgen, Atari
  runners/      replay, training loop, evaluation loop
src/configs/
  algo/ env/ method/ model/ sweep/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Optional benchmark environments:

```bash
pip install -e .[dmc]
pip install -e .[procgen]
pip install -e .[atari]
```

## Quick smoke test

```bash
python -m fatewm.experiments.toy_ablation.train \
  device=cpu \
  train.total_steps=200 \
  replay.warmup_steps=32 \
  train.batch_size=16 \
  eval.interval=100 \
  eval.episodes=3
```

## Main training entrypoints

Toy diagnostic:

```bash
python -m fatewm.experiments.toy_ablation.train
```

DMC:

```bash
python -m fatewm.experiments.dmc.train env=dmc_walker_walk method=scheme1_dmc
```

Procgen:

```bash
python -m fatewm.experiments.procgen.train env=procgen_coinrun method=scheme1_procgen
```

Atari-100k style discrete benchmark:

```bash
python -m fatewm.experiments.atari100k.train env=atari100k_breakout method=rrrm
```

## Key configuration knobs

- `timescales.deltas`: discretized horizons.
- `method.B`: horizon budget.
- `method.mediator`: `margin` or `listwise`.
- `method.risk_alpha`: CVaR tail level for horizon risk.
- `method.lambda_strong`: weight on budgeted strong correction.
- `method.lambda_router`: weight on router calibration.
- `method.teacher_ema`: EMA rate for the benchmark reference teacher.
- `behavior.name`: `random`, `algo`, `fixed`, or `interface`.

## Reproducibility notes

- All experiments are driven by Hydra configs.
- The code is intentionally written so that **the same candidate set** can be used by the model and by the oracle/teacher reference whenever available.
- For DMC and Procgen, oracle references are stored in replay extras and reused during updates.
- For environments where oracle references are unavailable, the code uses an EMA teacher to build the reference energy on the same candidate pool.

## Outputs

Hydra writes logs and configs to `outputs/YYYY-MM-DD/HH-MM-SS/`.

## Submission cleanup

This refactor removes the earlier unrelated front-end/app placeholder role of the repository and turns it into a paper-facing Python package focused on RRRM experiments.
