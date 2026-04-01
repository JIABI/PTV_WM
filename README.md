# Prediction-Target Validity Under Fixed Interfaces (PTV)

This repository is organized around the paper framework **Prediction-Target Validity (PTV) Under Fixed Interfaces**.

PTV is the top-level framing: it defines how prediction targets are specified, evaluated, and compared under a fixed deployment interface. The repository contains multiple executable instantiations of that framing.

## What PTV is

- **PTV is a framework** for target validity under fixed interfaces.
- **PTV is not a single new world-model architecture**.
- PTV is instantiated here by three separate model families:
  - **PTV-Regime** (LIC-WM lineage)
  - **PTV-Boundary** (RALAG-WM lineage)
  - **PTV-Criticality** (RRRM lineage)

## Repository Tree (paper-facing summary)

```text
.
├── ptv/
│   ├── core/            # shared framework-facing namespace
│   ├── regime/          # PTV-Regime integration namespace
│   ├── boundary/        # PTV-Boundary integration namespace
│   └── criticality/     # PTV-Criticality integration namespace
├── configs/
│   ├── shared/
│   ├── regime/
│   ├── boundary/
│   └── criticality/
├── scripts/             # top-level train/eval wrappers for the 3 instantiations
├── docs/
├── tests/
├── src/atlas_one_step/  # legacy atlas backend used as current shared-core implementation
├── PTV-Regime/          # LIC-WM implementation tree
├── PTV-Boundary/        # RALAG-WM implementation tree
└── PTV-Criticality/     # RRRM implementation tree
```

## Getting Started

### 1) Environment

At minimum (framework/core + docs + wrappers):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For instantiation-specific dependencies, also install from each subproject as needed:

- `PTV-Regime/requirements.txt`
- `PTV-Boundary` extras in `PTV-Boundary/pyproject.toml`
- `PTV-Criticality/requirements.txt` and optional benchmark extras

### 2) Quick sanity checks

```bash
python -c "import ptv, ptv.core, ptv.regime, ptv.boundary, ptv.criticality"
pytest -q tests/test_targets.py tests/test_smoke_pipeline.py
```

## Run Instructions by Instantiation

## PTV-Regime (LIC-WM)

Top-level wrappers:

```bash
python scripts/train_regime.py domain=lic_boids trainer.epochs=1 trainer.history_len=8 trainer.pred_len=4
python scripts/eval_regime.py evaluator=matched_geometry domain=lic_boids trainer.history_len=8 trainer.pred_len=4
```

Canonical implementation tree:
- Code: `PTV-Regime/src/licwm/`
- Configs: `PTV-Regime/configs/`
- Original scripts: `PTV-Regime/scripts/`

## PTV-Boundary (RALAG-WM)

Top-level wrappers:

```bash
python scripts/train_boundary.py env=dummy runtime.max_steps=1
python scripts/eval_boundary.py env=dummy
```

Canonical implementation tree:
- Code: `PTV-Boundary/src/ralagwm/`
- Configs: `PTV-Boundary/configs/`
- Original scripts: `PTV-Boundary/training/` and `PTV-Boundary/testing/`

## PTV-Criticality (RRRM)

Top-level wrappers:

```bash
python scripts/train_criticality.py device=cpu train.total_steps=200 replay.warmup_steps=32 train.batch_size=16
python scripts/eval_criticality.py env=toy
```

Canonical implementation tree:
- Code: `PTV-Criticality/src/fatewm/`
- Configs: `PTV-Criticality/src/configs/`
- Original experiment entrypoints: `PTV-Criticality/src/fatewm/experiments/*`

## Shared vs Instantiation-Specific Code

### Shared/framework-level
- `ptv/core/` provides the current shared framework-facing surface.
- `src/atlas_one_step/` is the current executable backend for shared fixed-interface target-selection pipeline pieces used in this migration phase.

### Instantiation-specific
- `ptv/regime` ↔ `PTV-Regime/src/licwm/`
- `ptv/boundary` ↔ `PTV-Boundary/src/ralagwm/`
- `ptv/criticality` ↔ `PTV-Criticality/src/fatewm/`

These remain explicitly separate; they are not collapsed into one model family.

## Expected Datasets, Configs, and Outputs

- **Framework/core smoke path** uses synthetic data via root `configs/*.yaml` and writes to root `outputs/`.
- **PTV-Regime** uses domain datasets/adapters documented under `PTV-Regime/data/` and writes under `PTV-Regime/outputs/`.
- **PTV-Boundary** uses manifests under `PTV-Boundary/inputs/manifests/` and writes under `PTV-Boundary/outputs/`.
- **PTV-Criticality** uses Hydra configs under `PTV-Criticality/src/configs/` and writes Hydra run outputs under `PTV-Criticality/outputs/`.

If optional external benchmarks are missing, use dummy/toy configs first.

## Paper-to-Code Structure

For detailed mapping, see `docs/paper_to_code_mapping.md`. In short:

- **Framework-level concepts** (fixed-interface setup, target-validity framing, operational selection interface, common evidence protocol) are documented and surfaced via `ptv/core` and current shared backend modules.
- **Executable instantiations** are provided by:
  - PTV-Regime (`licwm`)
  - PTV-Boundary (`ralagwm`)
  - PTV-Criticality (`fatewm`)

## Migration Notes

For users coming from earlier layouts and names:
- `docs/migration_from_old_layout.md`
- `docs/repo_map.md`

Known current-state note: this is a compatibility-first migration stage; canonical implementation trees still live in `PTV-Regime/`, `PTV-Boundary/`, `PTV-Criticality/`, and `src/atlas_one_step/` while top-level PTV namespaces and wrappers provide paper-aligned entry surfaces.
