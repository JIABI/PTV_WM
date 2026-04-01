# PTV-Regime

PTV-Regime is the **regime-focused instantiation** under the Prediction-Target Validity (PTV) framework.

> Naming alignment: this folder uses **PTV-Regime** as the primary name (PTV-Regime lineage).

## What this folder contains

- Regime-specific world-model code: `src/licwm/`
- Regime experiment configs: `configs/`
- Regime training/evaluation scripts: `scripts/`
- Regime data adapters and dataset notes: `data/`
- Regime outputs and paper artifacts: `outputs/`

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick run

```bash
python scripts/train.py domain=lic_boids trainer.epochs=1 trainer.history_len=8 trainer.pred_len=4
python scripts/evaluate.py evaluator=matched_geometry domain=lic_boids trainer.history_len=8 trainer.pred_len=4
python scripts/audit.py task=antisteg_audit domain=lic_boids trainer.history_len=8 trainer.pred_len=4
```

## Paper workflow

```bash
python scripts/run_matrix.py --manifest configs/sweep/main_section6.yaml
python scripts/aggregate_results.py
python scripts/make_main_tables.py
python scripts/make_main_figures.py
```

## Compatibility note

Internal module/config identifiers (for example `licwm`, `lic_boids`, and existing script/config filenames) are currently preserved for compatibility.
Public documentation and paper-facing naming in this repository should treat this instantiation as **PTV-Regime**.
