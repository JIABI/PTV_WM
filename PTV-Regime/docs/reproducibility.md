# Reproducibility
- Seed control: `utils/seed.py`
- Determinism toggle via config `deterministic`
- Environment logging: `training/engine.py -> env.json`
- Config snapshot: each run writes `config_snapshot.yaml`
- Checkpoint policy: `checkpoint_last.pt`, `checkpoint_best.pt`
- Metrics: `metrics.json`, `per_epoch_metrics.csv`, aggregate csv/json in `outputs/aggregates`
