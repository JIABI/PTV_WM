# Experiment matrix

This project uses a section-aligned experiment matrix. Each manifest expands into one or more train/eval/audit commands, and the resulting aggregated outputs are transformed into exact paper artifacts.

## Main sections
- Main Sec.5 setup/baselines/protocol:
  - manifest: `configs/sweep/main_section5.yaml`
  - aggregate: `outputs/aggregates/main_section5.csv`
  - table artifact: `outputs/tables/Table1_benchmark_protocol_metrics.*`
  - expected emphasis: benchmark family, evaluator path, metric protocol
- Main Sec.6 LIC-Boids existence + identifiability:
  - manifest: `configs/sweep/main_section6.yaml`
  - aggregate: `outputs/aggregates/main_section6.csv`
  - figure artifact: `outputs/figures/Figure2_lic_boids_mechanism.png`
  - table artifact: `outputs/tables/Table2_lic_boids_quantitative_summary.*`
  - expected emphasis: geometry match, law separation, response separation, monotonicity
- Main Sec.7 crowd + traffic cross-domain evidence:
  - manifest: `configs/sweep/main_section7.yaml`
  - aggregate: `outputs/aggregates/main_section7.csv`
  - figure artifact: `outputs/figures/Figure3_cross_domain_evidence.png`
  - table artifact: `outputs/tables/Table3_cross_domain_predictive_stability.*`
  - expected emphasis: step / rollout stability by domain and scene
- Main Sec.8 LIC-UAV control value:
  - manifest: `configs/sweep/main_section8.yaml`
  - aggregate: `outputs/aggregates/main_section8.csv`
  - figure artifact: `outputs/figures/Figure4_lic_uav_control_value.png`
  - table artifact: `outputs/tables/Table4_lic_uav_control_summary.*`
  - expected emphasis: success, safety, recovery, tail risk
- Main Sec.9 capacity / leakage / no-event / OOD:
  - manifest: `configs/sweep/main_section9.yaml`
  - aggregate: `outputs/aggregates/main_section9.csv`
  - figure artifact: `outputs/figures/Figure5_defense_stress_tests.png`
  - table artifact: `outputs/tables/Table5_ablation_stress_summary.*`
  - expected emphasis: TV / HF-ratio, no-event gap, OOD stress

## SI sections
- SI-A: `configs/sweep/si_a.yaml` -> `TableS1`, `TableS2`
- SI-B: `configs/sweep/si_b.yaml` -> `TableS3`, `TableS4`
- SI-C: `configs/sweep/si_c.yaml` -> `FigureS1`, `FigureS2`, `FigureS3`, `TableS5`
- SI-D: `configs/sweep/si_d.yaml` -> `FigureS4`, `FigureS5`, `TableS6`
- SI-E: `configs/sweep/si_e.yaml` -> `FigureS6`, `FigureS7`, `TableS7`
- SI-F: `configs/sweep/si_f.yaml` -> `FigureS8`, `FigureS9`, `TableS8`, `TableS9`
- SI-G: `configs/sweep/si_g.yaml` -> `FigureS10`, `FigureS11`, `FigureS12`, `TableS10`
- SI-H: `configs/sweep/si_h.yaml` -> `FigureS13`, `FigureS14`, `FigureS15`, `TableS11`
- SI-I: `configs/sweep/si_i.yaml` -> `FigureS16`, `FigureS17`, `FigureS18`, `TableS12`

## Execution
Run a section manifest:
```bash
python scripts/run_matrix.py --manifest configs/sweep/main_section6.yaml
```
Aggregate and export exact paper artifacts:
```bash
python scripts/aggregate_results.py
python scripts/export_paper_artifacts.py
```
Validate artifact readiness:
```bash
cat outputs/tables/artifact_validation_report.json
```
