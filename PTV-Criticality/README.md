# PTV-Criticality

PTV-Criticality is the **criticality-focused instantiation** under the Prediction-Target Validity (PTV) framework.

> Naming alignment: this folder uses **PTV-Criticality** as the primary name (PTV-Criticality lineage).

## What this folder contains

- Criticality-specific algorithm/core code: `src/fatewm/`
- Criticality Hydra configs: `src/configs/`
- Benchmark scripts: `scripts/`
- Run guidance: `RUN_BENCHMARKS.md`, `RUN_DMC.md`

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick run

```bash
python -m fatewm.experiments.toy_ablation.train device=cpu train.total_steps=200 replay.warmup_steps=32 train.batch_size=16
python -m fatewm.experiments.toy_ablation.eval env=toy
```

## Benchmark examples

```bash
python -m fatewm.experiments.dmc.train env=dmc_walker_walk method=scheme1_dmc
python -m fatewm.experiments.procgen.train env=procgen_coinrun method=scheme1_procgen
python -m fatewm.experiments.atari100k.train env=atari100k_breakout method=ptv_criticality
```

## Compatibility note

Internal module/config identifiers (for example `fatewm` and existing method filenames) are currently preserved for compatibility aliases.
Public documentation and paper-facing naming in this repository should treat this instantiation as **PTV-Criticality**.
