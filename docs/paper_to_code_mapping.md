# Paper-to-Code Mapping

This document maps paper concepts in **Prediction-Target Validity Under Fixed Interfaces** to current code.

## Scope note

Some concepts are currently represented as **framework-level documentation + integration surfaces**, while executable implementations remain in existing canonical trees from previous codebases. This is intentional for compatibility.

## 1) Fixed-interface deployment setup

**Paper concept:** evaluate prediction targets under a fixed downstream interface rather than changing interface semantics per target.

**Code mapping:**
- Framework-facing namespace: `ptv/core/`.
- Current executable backend: `src/atlas_one_step/` modules and CLI (`src/atlas_one_step/cli.py`).
- Instantiation execution surfaces:
  - `scripts/train_regime.py`, `scripts/eval_regime.py`
  - `scripts/train_boundary.py`, `scripts/eval_boundary.py`
  - `scripts/train_criticality.py`, `scripts/eval_criticality.py`

## 2) Target-validity criteria

**Paper concept:** validity criteria for assessing whether target choices preserve operational usefulness under fixed interfaces.

**Code mapping:**
- Framework-level target/metric pipeline (current backend): `src/atlas_one_step/targets.py`, `src/atlas_one_step/metrics.py`, `src/atlas_one_step/selection.py`.
- Instantiation-specific evaluation code:
  - Regime: `PTV-Regime/src/licwm/evaluation/`, `PTV-Regime/src/licwm/metrics/`, `PTV-Regime/src/licwm/audits/`
  - Boundary: `PTV-Boundary/src/ralagwm/evaluation/`, `PTV-Boundary/src/ralagwm/audit/`, `PTV-Boundary/src/ralagwm/geometry/metrics.py`
  - Criticality: `PTV-Criticality/src/fatewm/core/metrics.py`, runners and experiment eval modules

## 3) Operational target-selection interface

**Paper concept:** an operational interface for selecting and deploying targets without redefining the consumer interface.

**Code mapping:**
- Framework/core operational selection path: atlas CLI flow in `src/atlas_one_step/cli.py` (`run-sweep`, `build-atlas`, `fit-surrogate`, `select-target`, `train`, `evaluate`).
- Wrapper-level operational entrypoints by instantiation: `scripts/train_*` and `scripts/eval_*`.

## 4) Common evidence protocol

**Paper concept:** comparable evidence generation across candidate targets/instantiations.

**Code mapping:**
- Shared/core smoke protocol and tests: `tests/test_smoke_pipeline.py`, `tests/test_targets.py`.
- Per-instantiation experiment/evaluation pipelines:
  - Regime: `PTV-Regime/scripts/`, section manifests under `PTV-Regime/configs/sweep/`
  - Boundary: `PTV-Boundary/training/`, `PTV-Boundary/testing/`, manifests under `PTV-Boundary/inputs/manifests/`
  - Criticality: `PTV-Criticality/src/fatewm/experiments/`, sweeps under `PTV-Criticality/src/configs/sweep/`

## 5) PTV-Regime

- Paper alignment: regime-oriented instantiation (PTV-Regime lineage).
- Executable code: `PTV-Regime/src/licwm/`.
- PTV namespace surface: `ptv/regime`.
- Top-level run wrappers: `scripts/train_regime.py`, `scripts/eval_regime.py`.

## 6) PTV-Boundary

- Paper alignment: boundary-oriented instantiation (PTV-Boundary lineage).
- Executable code: `PTV-Boundary/src/ralagwm/`.
- PTV namespace surface: `ptv/boundary`.
- Top-level run wrappers: `scripts/train_boundary.py`, `scripts/eval_boundary.py`.

## 7) PTV-Criticality

- Paper alignment: criticality-oriented instantiation (PTV-Criticality lineage).
- Executable code: `PTV-Criticality/src/fatewm/`.
- PTV namespace surface: `ptv/criticality`.
- Top-level run wrappers: `scripts/train_criticality.py`, `scripts/eval_criticality.py`.

## Framework-level vs executable implementation boundaries

- **Framework-level only in current phase:** top-level framing, shared namespace surfaces under `ptv/*`, normalized docs and wrapper entrypoints.
- **Executable model/training/eval code:** remains in canonical implementation trees (`src/atlas_one_step`, `PTV-Regime`, `PTV-Boundary`, `PTV-Criticality`).

## TODO

- Further consolidation so canonical executable code physically lives under `ptv/{core,regime,boundary,criticality}` without bridge imports.
