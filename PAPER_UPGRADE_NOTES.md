# Paper-Level Upgrade Notes

## NCS manuscript upgrade (current)

This repository now includes a source-data-first reproducibility layer for the NCS molecular-screening MLIP hero case.

Added:
- `src/ptv_ncs_mlip/` package for archive validation, round metrics, target nomination, stats, tables, figures, and manifest export.
- `configs/ncs_mlip_hero.yaml` as the primary NCS config.
- Pipeline scripts:
  - `run_ncs_mlip_pipeline.sh` (primary)
  - `run_legacy_atlas_pipeline.sh` (legacy compatibility rename)

Compatibility note: legacy ATLAS/stress-test code remains in place and is retained for historical lineage comparisons.
