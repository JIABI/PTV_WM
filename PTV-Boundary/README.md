# PTV-Boundary

PTV-Boundary is the **boundary-focused instantiation** under the Prediction-Target Validity (PTV) framework.

> Naming alignment: this folder uses **PTV-Boundary** as the primary name (PTV-Boundary lineage).

## What this folder contains

- Boundary-specific model/evaluation code: `src/ralagwm/`
- Boundary configs: `configs/`
- Boundary training entrypoints: `training/`
- Boundary evaluation entrypoints: `testing/`
- Domain manifests: `inputs/manifests/`
- Boundary outputs: `outputs/`

## Install

```bash
python -m pip install -e .[dev]
```

## Quick run

```bash
python training/train_ralag_wm.py env=dummy runtime.max_steps=1
python testing/eval_main_benchmark.py env=dummy
```

## Paper workflow

```bash
python training/run_paper_training.py
python testing/run_paper_main.py
python testing/run_paper_si.py
```

## Compatibility note

Internal module/config identifiers (for example `ralagwm`, `ralag_wm`, and existing script/config filenames) are currently preserved for compatibility.
Public documentation and paper-facing naming in this repository should treat this instantiation as **PTV-Boundary**.
