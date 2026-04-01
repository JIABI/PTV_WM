# Paper artifact manifest

This document fixes the exact artifact names, expected aggregate sources, preferred fields, and caption stubs produced by the codebase for the paper results section.

## Main paper

### Figure 1
- artifact: `outputs/figures/Figure1_method_overview.txt`
- generated: manual
- caption stub: `outputs/figures/Figure1_method_overview_caption.txt`
- role: method overview schematic aligned to Section 4

### Figure 2
- artifact: `outputs/figures/Figure2_lic_boids_mechanism.png`
- figure-ready CSV: `outputs/figures/figure_ready/Figure2_lic_boids_mechanism.csv`
- aggregate source: `outputs/aggregates/main_section6.csv`
- preferred fields: `geometry_match_quality`, `law_separation`, `response_separation`, `monotonicity`
- role: existence + identifiability + counterfactual evidence on LIC-Boids

### Figure 3
- artifact: `outputs/figures/Figure3_cross_domain_evidence.png`
- figure-ready CSV: `outputs/figures/figure_ready/Figure3_cross_domain_evidence.csv`
- aggregate source: `outputs/aggregates/main_section7.csv`
- preferred fields: `step_rmse`, `rollout_ade`, `rollout_fde`, `rollout_horizon`
- role: crowd + traffic cross-domain predictive stability

### Figure 4
- artifact: `outputs/figures/Figure4_lic_uav_control_value.png`
- figure-ready CSV: `outputs/figures/figure_ready/Figure4_lic_uav_control_value.csv`
- aggregate source: `outputs/aggregates/main_section8.csv`
- preferred fields: `success_rate`, `safety_violation`, `recovery_latency`, `cvar`
- role: LIC-UAV control value

### Figure 5
- artifact: `outputs/figures/Figure5_defense_stress_tests.png`
- figure-ready CSV: `outputs/figures/figure_ready/Figure5_defense_stress_tests.csv`
- aggregate source: `outputs/aggregates/main_section9.csv`
- preferred fields: `tv_law`, `hf_ratio`, `scale_transfer_gap`, `rollout_horizon`
- role: capacity / leakage / no-event / OOD defense

### Tables 1--5
- Table 1: `outputs/tables/Table1_benchmark_protocol_metrics.csv`
- Table 2: `outputs/tables/Table2_lic_boids_quantitative_summary.csv`
- Table 3: `outputs/tables/Table3_cross_domain_predictive_stability.csv`
- Table 4: `outputs/tables/Table4_lic_uav_control_summary.csv`
- Table 5: `outputs/tables/Table5_ablation_stress_summary.csv`
- each table also exports:
  - markdown preview: `*.md`
  - metadata json: `*.json`
  - caption stub: `*_caption.txt`

## Supplementary figures and tables

Each SI artifact follows the same convention:
- plot/table artifact with fixed filename
- source aggregate in `outputs/aggregates/si_*.csv`
- metadata json
- caption stub txt
- figure-ready CSV for generated figures

### SI mapping
- SI-A -> `si_a.csv` -> `TableS1`, `TableS2`
- SI-B -> `si_b.csv` -> `TableS3`, `TableS4`
- SI-C -> `si_c.csv` -> `FigureS1`, `FigureS2`, `FigureS3`, `TableS5`
- SI-D -> `si_d.csv` -> `FigureS4`, `FigureS5`, `TableS6`
- SI-E -> `si_e.csv` -> `FigureS6`, `FigureS7`, `TableS7`
- SI-F -> `si_f.csv` -> `FigureS8`, `FigureS9`, `TableS8`, `TableS9`
- SI-G -> `si_g.csv` -> `FigureS10`, `FigureS11`, `FigureS12`, `TableS10`
- SI-H -> `si_h.csv` -> `FigureS13`, `FigureS14`, `FigureS15`, `TableS11`
- SI-I -> `si_i.csv` -> `FigureS16`, `FigureS17`, `FigureS18`, `TableS12`

## Validation

Running

```bash
python scripts/export_paper_artifacts.py
```

also writes

- `outputs/tables/artifact_validation_report.json`

which reports whether each artifact source exists and whether required fields are present.


## Machine-readable manifest

Running `python scripts/export_paper_artifacts.py` also writes:

- `outputs/tables/paper_artifact_index.json`

which records the fixed artifact ids, source CSVs, preferred columns, required columns, and caption stubs for every Main/SI figure and table.
