install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest -q

smoke:
	python -m atlas_one_step.cli smoke-test --config configs/default_smoke.yaml


ncs-mlip:
	python -m ptv_ncs_mlip.cli export-all --config configs/ncs_mlip_hero.yaml

validate-ncs:
	python -m ptv_ncs_mlip.cli validate-archive --data-root data/ncs_mlip

test-ncs:
	pytest -q tests/test_ncs_mlip_*.py
