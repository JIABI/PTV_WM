#!/usr/bin/env bash
set -euo pipefail

python -m atlas_one_step.cli run-sweep --config configs/cifar10_line_paper.yaml
python -m atlas_one_step.cli build-atlas --config configs/cifar10_line_paper.yaml
python -m atlas_one_step.cli fit-surrogate --atlas outputs/cifar10_line_paper/atlas/atlas.parquet
python analysis/paper_results.py \
  --summary-root outputs/cifar10_line_paper \
  --atlas outputs/cifar10_line_paper/atlas/atlas.parquet \
  --outdir outputs/cifar10_line_paper/analysis
