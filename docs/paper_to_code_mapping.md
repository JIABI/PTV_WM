# Paper-to-Code Mapping

## Primary NCS reproducibility path (current manuscript)

The primary executable path for the NCS manuscript molecular-screening hero case is:

- Package: `src/ptv_ncs_mlip/`
- Config: `configs/ncs_mlip_hero.yaml`
- Archive input root: `data/ncs_mlip/`
- Export entrypoint: `python -m ptv_ncs_mlip.cli export-all --config configs/ncs_mlip_hero.yaml`

Outputs include:
- Fig. 2 source-data exports and figures
- Extended Data Table 1 CSV/TeX
- Supplementary statistics table CSV/TeX and `si_mlip_stats.csv`
- `ptv_nomination.json`
- `manifest.json`

## Legacy compatibility / stress-test lineages

Legacy and stress-test trees are retained:
- `src/atlas_one_step/` (legacy migration backend)
- `PTV-Regime/` (LIC-WM lineage)
- `PTV-Boundary/` (RALAG-WM lineage)
- `PTV-Criticality/` (RRRM lineage)

Compatibility note: these trees are preserved and remain runnable, but are not the primary NCS molecular-screening reproducibility layer.
