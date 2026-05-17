# Prediction-Target Validity Under Fixed Interfaces (PTV)

This repository supports the NCS manuscript: **A prospective protocol for reliable closed-loop deployment of computational models.**

The primary reproducible case is the **molecular-screening MLIP hero case**.
Legacy **PTV-Regime / PTV-Boundary / PTV-Criticality** trees are retained as stress-test instantiations and historical implementation lineages.

## NCS reproducibility (primary path)

```bash
pip install -e .
python -m ptv_ncs_mlip.cli export-all --config configs/ncs_mlip_hero.yaml
```

Expected outputs under `outputs/ncs_mlip/`:
- Extended Data Table 1 source data
- Supplementary test-family table
- Fig. 2 source data
- PTV nomination JSON
- archive manifest

## Repository map (NCS package)
- `src/ptv_ncs_mlip/` — NCS molecular-screening reproducibility layer
- `configs/ncs_mlip_hero.yaml` — hero-case config
- `data/ncs_mlip/` — frozen source-data archive inputs
- `outputs/ncs_mlip/` — generated reproducibility artifacts

## Legacy / stress-test instantiations

Legacy compatibility code remains available and is not the primary NCS reproduction path:
- `src/atlas_one_step/` (legacy ATLAS migration backend)
- `PTV-Regime/` (LIC-WM lineage)
- `PTV-Boundary/` (RALAG-WM lineage)
- `PTV-Criticality/` (RRRM lineage)
