# DMC notes

The DMC pipeline uses a **candidate-based energy interface** and an **auditable oracle reference** built from simulator state snapshots.

## Quick run

```bash
python -m fatewm.experiments.dmc.train \
  env=dmc_walker_walk \
  method=scheme1_dmc \
  algo=minimal \
  device=cuda
```

## Useful overrides

- `method.num_candidates=32`
- `method.oracle_horizon=1`
- `method.candidate_sigma=0.30`
- `interface.refine.enabled=true`
- `interface.refine.method=mirror`
- `eval.policy=interface`

## Debug configuration

```bash
python -m fatewm.experiments.dmc.train \
  env=dmc_walker_walk \
  method=scheme1_dmc \
  train.total_steps=2000 \
  replay.warmup_steps=200 \
  eval.interval=200 \
  eval.episodes=3 \
  device=cpu
```
