# Repository Map

This document describes the current repository tree after the PTV-first refactor pass, and marks which directories are framework-shared vs instantiation-specific.

## Reader-friendly tree

```text
.
├── README.md
├── ptv/
│   ├── __init__.py
│   ├── core/                    [Shared framework-facing namespace]
│   ├── regime/                  [Instantiation bridge: PTV-Regime]
│   ├── boundary/                [Instantiation bridge: PTV-Boundary]
│   └── criticality/             [Instantiation bridge: PTV-Criticality]
├── configs/
│   ├── *.yaml                   [Current shared/core smoke + atlas configs]
│   ├── shared/                  [Normalized shared-config anchor]
│   ├── regime/                  [Normalized regime-config anchor]
│   ├── boundary/                [Normalized boundary-config anchor]
│   └── criticality/             [Normalized criticality-config anchor]
├── scripts/                     [Top-level runnable wrappers]
├── docs/
├── tests/                       [Root framework smoke tests]
├── src/atlas_one_step/          [Shared-core executable backend in this phase]
├── PTV-Regime/                  [Instantiation-specific canonical tree]
├── PTV-Boundary/                [Instantiation-specific canonical tree]
└── PTV-Criticality/             [Instantiation-specific canonical tree]
```

## Directory purposes

## Shared framework code and interfaces
- `ptv/core/`
  - Framework-facing import surface for shared target-validity pipeline entrypoints.
- `src/atlas_one_step/`
  - Current executable backend for shared fixed-interface target-selection flow.
- `tests/`
  - Root-level smoke tests validating shared/core behavior.

## Instantiation-specific code
- `PTV-Regime/`
  - PTV-Regime implementation, configs, scripts, datasets, outputs.
- `PTV-Boundary/`
  - PTV-Boundary implementation, configs, train/eval scripts, manifests, outputs.
- `PTV-Criticality/`
  - PTV-Criticality implementation, Hydra experiments, configs, outputs.

## Top-level integration surfaces
- `ptv/regime`, `ptv/boundary`, `ptv/criticality`
  - Integration namespaces that expose the three instantiations under the PTV framing.
- `scripts/`
  - Root train/eval wrappers for the three instantiations.

## Config organization status

Normalized top-level config folders exist under `configs/{shared,regime,boundary,criticality}`. In this compatibility phase, canonical executable config trees still live in:
- `PTV-Regime/configs/`
- `PTV-Boundary/configs/`
- `PTV-Criticality/src/configs/`

## TODO (structure convergence)

- Consolidate canonical per-instantiation config trees directly under `configs/regime`, `configs/boundary`, and `configs/criticality`.
- Gradually move bridge-based integration namespaces to direct in-tree package ownership under `ptv/*`.
