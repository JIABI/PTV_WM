# Paper-Level Upgrade Notes

## What changed in this revision

This revision aligns the codebase more tightly with the current paper draft.

- Added paper-facing configs for the main experimental roles:
  - CIFAR-10 compact atlas sweeps
  - ImageNet-64 atlas-guided method validation
  - LSUN Bedroom-256 strongest-instantiation stress testing
  - PDEBench Darcy transfer
  - UCF101 frame pilot
- Reworked `fit_surrogates` to export stronger diagnostics-vs-label metrics, including cross-validated phase and failure-prediction summaries.
- Reworked target selection to emit a richer selected-target artifact with objective decomposition:
  - predicted pathology
  - semantic gap
  - complexity penalty
  - trainable / rescue / failure probabilities
- Replaced brittle local-path `run.sh` and `run_multi.sh` scripts with repository-relative scripts.
- Added `analysis/paper_results.py` to export flat summary tables and plot-ready CSV/PNG artifacts for the paper.
- Updated `README.md` so the release instructions match the current paper structure.

## Recommended entry points

### Compact atlas construction
```bash
python -m atlas_one_step.cli run-sweep --config configs/cifar10_line_paper.yaml
python -m atlas_one_step.cli build-atlas --config configs/cifar10_line_paper.yaml
python -m atlas_one_step.cli fit-surrogate --atlas outputs/cifar10_line_paper/atlas/atlas.parquet
```

### Atlas-guided target selection
```bash
python -m atlas_one_step.cli select-target \
  --atlas outputs/cifar10_line_paper/atlas/atlas.parquet \
  --surrogate outputs/cifar10_line_paper/atlas/surrogate.joblib \
  --output outputs/cifar10_line_paper/atlas/selected_target.json \
  --config configs/imagenet64_train_paper.yaml
```

### Method comparison
```bash
python -m atlas_one_step.cli train --config configs/imagenet64_train_paper.yaml --mode coupled
python -m atlas_one_step.cli train \
  --config configs/imagenet64_train_paper.yaml \
  --mode atlas_guided \
  --selected-target outputs/cifar10_line_paper/atlas/selected_target.json
```

### Plot-ready exports
```bash
python analysis/paper_results.py \
  --summary-root outputs \
  --atlas outputs/cifar10_line_paper/atlas/atlas.parquet \
  --outdir outputs/paper_results
```

## Validation notes

- Existing smoke tests still pass.
- `tiny_unet` remains supported for smoke/debug mode.
- `paper_unet` and the paper configs remain CPU-runnable for small checks, while the larger configs are intended for real datasets and longer budgets.
