# ATLAS: Target-Pathology Atlas for One-Step High-Dimensional Continuous Generation

This repository implements the paper-aligned ATLAS pipeline for studying **target-induced inverse problems** in one-step generation. It includes:

- structured target families: line, simplex, and scheduled targets
- a fixed diffusion-like corruption process
- pathology probes: support deviation, normal burden, conditioning, and early-training instability
- atlas construction and surrogate fitting
- atlas-guided target selection
- coupled, unguided decoupled, and atlas-guided decoupled training
- plot-ready exports for the paper’s main tables and figures

The code is runnable end-to-end on a synthetic smoke dataset and is organized so the same CLI can be used for:

- **CIFAR-10 / 32x32** compact atlas construction
- **ImageNet-64 / 64x64** atlas-guided method validation
- **LSUN Bedroom-256 / 256x256** hard-regime and tail-failure stress tests
- **PDEBench Darcy** transfer to scientific continuous fields
- **UCF101 frame pilot** supplementary temporal validation

These roles mirror the paper’s experimental program, which separates dense atlas construction, medium-scale validation, high-resolution stress testing, and cross-domain transfer fileciteturn1file0L214-L231.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Smoke test

```bash
python -m atlas_one_step.cli smoke-test --config configs/default_smoke.yaml
```

This runs:
1. a small target sweep,
2. pathology computation,
3. atlas build,
4. surrogate fitting,
5. target selection,
6. one short coupled training,
7. one short atlas-guided decoupled training,
8. evaluation.

## Paper-aligned entry points

### Compact CIFAR-10 atlas
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

### ImageNet-64 coupled vs atlas-guided training
```bash
python -m atlas_one_step.cli train --config configs/imagenet64_train_paper.yaml --mode coupled
python -m atlas_one_step.cli train \
  --config configs/imagenet64_train_paper.yaml \
  --mode atlas_guided \
  --selected-target outputs/cifar10_line_paper/atlas/selected_target.json
```

### Plot-ready paper exports
```bash
python analysis/paper_results.py \
  --summary-root outputs \
  --atlas outputs/cifar10_line_paper/atlas/atlas.parquet \
  --outdir outputs/paper_results
```

### One-command convenience script
```bash
./run_paper_pipeline.sh
```

## Configuration files

- `configs/cifar10_line_paper.yaml`: line-family sweep for the main phase-band result
- `configs/cifar10_boundary_refine.yaml`: boundary-refined sweep
- `configs/cifar10_simplex_paper.yaml`: simplex sweep for higher-dimensional phase structure
- `configs/cifar10_train_paper.yaml`: compact paper-strength training config
- `configs/imagenet64_train_paper.yaml`: medium-scale method validation
- `configs/lsun256_strongest_paper.yaml`: high-resolution stress-test config
- `configs/darcy_transfer_paper.yaml`: scientific-field transfer config
- `configs/ucf101_pilot_paper.yaml`: supplementary video-frame pilot config

## Repository layout

- `configs/`: experiment YAMLs
- `src/atlas_one_step/`: package code
- `analysis/`: plot-ready exports and aggregation scripts
- `tests/`: smoke and target round-trip tests
- `outputs/`: generated runs

## Notes on scope

The main method in the paper has three stages—build atlas, select prediction target, and train decoupled model—and the code mirrors that structure directly. The target family, atlas, and decoupled objective are implemented to match the paper’s method section.
