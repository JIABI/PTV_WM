# Migration from Old Layout to PTV-First Layout

This guide maps prior project names/paths to the current paper-aligned PTV framing.

## Naming migration

| Old naming | Current naming |
|---|---|
| atlas / atlas-one-step (top-level identity) | PTV framework-level shared/core layer |
| PTV-Regime | PTV-Regime |
| PTV-Boundary | PTV-Boundary |
| PTV-Criticality / PTV-Criticality | PTV-Criticality |

## Path migration (old -> current)

## Framework/shared

| Old path | Current path | Notes |
|---|---|---|
| `src/atlas_one_step/*` | `ptv/core` surface + `src/atlas_one_step/*` backend | Shared executable backend retained in current phase |
| Root atlas docs as primary identity | `README.md` PTV-first framing + docs in `docs/` | Atlas is no longer repository identity |

## Regime

| Old path | Current path | Notes |
|---|---|---|
| `PTV-Regime/src/licwm/*` | Canonical executable remains same; exposed via `ptv/regime` | Bridge namespace added |
| `PTV-Regime/scripts/train.py` | `scripts/train_regime.py` | Wrapper delegates to canonical script |
| `PTV-Regime/scripts/evaluate.py` | `scripts/eval_regime.py` | Wrapper delegates to canonical script |

## Boundary

| Old path | Current path | Notes |
|---|---|---|
| `PTV-Boundary/src/ralagwm/*` | Canonical executable remains same; exposed via `ptv/boundary` | Bridge namespace added |
| `PTV-Boundary/training/train_ralag_wm.py` | `scripts/train_boundary.py` | Wrapper delegates to canonical script |
| `PTV-Boundary/testing/eval_main_benchmark.py` | `scripts/eval_boundary.py` | Wrapper delegates to canonical script |

## Criticality

| Old path | Current path | Notes |
|---|---|---|
| `PTV-Criticality/src/fatewm/*` | Canonical executable remains same; exposed via `ptv/criticality` | Bridge namespace added |
| `python -m fatewm.experiments...` from subproject | `scripts/train_criticality.py` / `scripts/eval_criticality.py` wrappers at root | Module path kept intact under subproject |

## Config path normalization status

Normalized anchors now exist at:
- `configs/shared/`
- `configs/regime/`
- `configs/boundary/`
- `configs/criticality/`

Current canonical executable config trees still live at:
- `PTV-Regime/configs/`
- `PTV-Boundary/configs/`
- `PTV-Criticality/src/configs/`

## How old projects map into PTV framing

- **Atlas code** is kept where it supports shared framework/core behavior under fixed interfaces; it is no longer the repository's dominant identity.
- **PTV-Regime** is treated as the executable basis for **PTV-Regime**.
- **PTV-Boundary** is treated as the executable basis for **PTV-Boundary**.
- **PTV-Criticality** is treated as the executable basis for **PTV-Criticality**.

## Known compatibility notes

1. This is a compatibility-first phase; several integrations are bridge-based (`ptv/*` namespace wrappers and root `scripts/*` wrappers).
2. Per-instantiation canonical code/config trees are not yet fully physically relocated under `ptv/*` and `configs/*`.
3. Existing subproject-native commands remain valid and are still the canonical low-level run paths.
4. If optional benchmark dependencies are unavailable, use dummy/toy settings first.
