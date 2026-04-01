.PHONY: test lint smoke

test:
	pytest -q

smoke:
	python training/train_audit.py env=dummy runtime.max_steps=2
	python training/train_ralag_wm.py env=dummy runtime.max_steps=2
	python testing/eval_main_benchmark.py env=dummy
