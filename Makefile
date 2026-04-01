install:
	pip install -r requirements.txt && pip install -e .

test:
	pytest -q

smoke:
	python -m atlas_one_step.cli smoke-test --config configs/default_smoke.yaml
