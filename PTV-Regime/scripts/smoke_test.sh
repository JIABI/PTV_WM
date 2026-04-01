#!/usr/bin/env bash
set -e
pip install -e . --no-build-isolation
python -c "import licwm; print('import_ok')"
python scripts/train.py trainer.epochs=1 trainer.batch_size=2 trainer.history_len=6 trainer.pred_len=3 domain=lic_boids domain.n_samples=8 domain.num_agents=8 output_dir=smoke_run
python scripts/evaluate.py evaluator=matched_geometry trainer.history_len=6 trainer.pred_len=3 domain=lic_boids domain.n_samples=8 domain.num_agents=8 checkpoint_path=smoke_run/checkpoint_best.pt
python scripts/audit.py task=antisteg_audit trainer.history_len=6 trainer.pred_len=3 domain=lic_boids domain.n_samples=8 domain.num_agents=8 checkpoint_path=smoke_run/checkpoint_best.pt
PYTHONPATH=src pytest -q src/licwm/tests/test_model_forward.py src/licwm/tests/test_jump_modes.py
python scripts/export_paper_artifacts.py || true
