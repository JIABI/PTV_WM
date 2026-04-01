# AGENTS.md (top-level guidance)

## Repository framing
- Treat this repository as **PTV-first** (Prediction-Target Validity Under Fixed Interfaces).
- Do **not** present atlas as the primary identity.
- Do **not** describe PTV as a fourth model family.

## Instantiation naming
- Use paper-aligned names consistently:
  - PTV-Regime (LIC-WM lineage)
  - PTV-Boundary (RALAG-WM lineage)
  - PTV-Criticality (RRRM lineage)

## Refactor policy
- Prefer incremental move/rename/restructure over rewrites.
- Preserve algorithmic behavior unless integration or a verified bug requires change.
- Keep instantiations explicitly separate; only share genuinely common framework logic.

## Documentation policy
- README and docs must stay paper-facing and map concepts to real paths/entrypoints.
- If something is not fully consolidated yet, mark it as TODO/compatibility note rather than implying completion.
