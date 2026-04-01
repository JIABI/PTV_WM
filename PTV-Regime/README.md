# LIC-WM (paper-level framework)

LIC-WM implements **law-state factorization** for long-horizon multi-agent world models:
- history encoder `q_phi(c_t|H_t)`
- law head `T_psi(c_t) -> (rho,beta,tau)`
- time-shared channelized prototype bank
- law-shaped message passing
- fast state update + slow/jump climate dynamics

## Install
```bash
pip install -r requirements.txt
pip install -e .
```

## Tiny run
```bash
python scripts/train.py domain=lic_boids trainer.epochs=1 trainer.history_len=8 trainer.pred_len=4
python scripts/evaluate.py evaluator=matched_geometry domain=lic_boids trainer.history_len=8 trainer.pred_len=4
python scripts/evaluate.py evaluator=counterfactual domain=lic_boids trainer.history_len=8 trainer.pred_len=4
python scripts/audit.py task=antisteg_audit domain=lic_boids trainer.history_len=8 trainer.pred_len=4
python scripts/aggregate_results.py
python scripts/make_main_tables.py && python scripts/make_main_figures.py
```

## Reproduction tiers

This release supports three levels of use:
1. **Smoke / fallback mode**: fully runnable from the bundled code and tiny synthetic adapters.
2. **Framework-level reproduction**: runnable experiment manifests, evaluators, audits, and artifact export on synthetic or adapter-backed data.
3. **Full paper benchmark reproduction**: requires the external datasets and simulator assets described under `data/` and in `docs/reproducibility.md`.

## Data adapters
- ETH/UCY, SDD, INTERACTION, and LIC-UAV adapters support **external-data mode** and **fallback/mock mode**.
- Full paper reproduction requires external assets under `data/ETH_UCY/`, `data/SDD/`, `data/INTERACTION/`, and `data/LIC_UAV/`.
- The repository does **not** redistribute those third-party datasets or simulator assets.

## Outputs
- `outputs/runs/<run>/`: checkpoints, metrics, config snapshot, env log
- `outputs/aggregates/`: section-level csv/json
- `outputs/tables/`: markdown tables
- `outputs/figures/`: plotted figures



## Paper section alignment
Run an entire paper section manifest with:
```bash
python scripts/run_matrix.py --manifest configs/sweep/main_section6.yaml
```

Available manifests:
- `configs/sweep/main_section5.yaml` ... `main_section9.yaml`
- `configs/sweep/si_a.yaml` ... `si_i.yaml`

After runs complete:
```bash
python scripts/aggregate_results.py
python scripts/make_main_tables.py
python scripts/make_main_figures.py
python scripts/make_si_tables.py
python scripts/make_si_figures.py
```
The resulting section-aligned artifacts are written under `outputs/aggregates`, `outputs/tables`, and `outputs/figures`.


## Exact paper artifact export
After running one or more section manifests and aggregating results, export the exact paper-named artifacts with:
```bash
python scripts/export_paper_artifacts.py
```
This writes:
- exact main table names under `outputs/tables/`
- exact main/SI figure names under `outputs/figures/`
- figure-ready CSV files under `outputs/figures/figure_ready/`
- manual/non-generated artifact notes (e.g. Figure 1 schematic)

See `docs/paper_artifact_manifest.md` for the fixed mapping between paper Figure/Table numbers and code outputs.


See `docs/final_display_contract.md` for the fixed presentation-layer ordering and naming contract used by exported paper artifacts.


## Release note
This supplementary release is intended as a runnable LIC-WM framework, synthetic fallback path, and paper-artifact generation package. Full reproduction of all reported benchmark results additionally requires the external datasets and simulator assets listed above.

## Checkpoint bootstrap

Evaluation and audit entry points will automatically bootstrap a minimal training run if no checkpoint is available and `auto_train_if_missing_checkpoint=true` (the default).

## Run by paper subsection
You can run experiments in the same order as the paper subsection layout. Each shell script runs its subsection manifest, aggregates results, and exports paper artifacts.

Main paper order:
```bash
bash scripts/paper/01_main_section5_setup.sh
bash scripts/paper/02_main_section6_lic_boids.sh
bash scripts/paper/03_main_section7_cross_domain.sh
bash scripts/paper/04_main_section8_lic_uav.sh
bash scripts/paper/05_main_section9_defense.sh
```

Or run all main-paper subsections in order:
```bash
bash scripts/paper/run_main_in_order.sh
```

Supplementary order:
```bash
bash scripts/paper/10_si_a_datasets.sh
bash scripts/paper/11_si_b_baselines.sh
bash scripts/paper/12_si_c_lic_boids_extended.sh
bash scripts/paper/13_si_d_counterfactual.sh
bash scripts/paper/14_si_e_cross_domain_extended.sh
bash scripts/paper/15_si_f_uav_extended.sh
bash scripts/paper/16_si_g_capacity_residual.sh
bash scripts/paper/17_si_h_antisteg_leakage.sh
bash scripts/paper/18_si_i_event_ood.sh
```

Or run everything in order:
```bash
bash scripts/paper/run_all_in_order.sh
```

Set `PYTHON_BIN=/path/to/python` if needed.


## Aggregation behavior

The paper subsection scripts now aggregate only the current subsection by default, so running `bash scripts/paper/02_main_section6_lic_boids.sh` will not print warnings for unrelated sections such as `main_section8` or `si_i`. To aggregate every section explicitly, run `python scripts/aggregate_results.py --verbose`.

## Recommended paper-level defaults

The default trainer configuration is now paper-oriented (`epochs=100`, `batch_size=64`, `history_len=16`, `pred_len=12`). The smoke and bootstrap paths continue to override these values down to tiny settings for quick checks.
