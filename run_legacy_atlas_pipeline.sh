#!/usr/bin/env bash
set -euo pipefail

# Compact atlas construction on CIFAR-10 / 32x32
python -m atlas_one_step.cli run-sweep --config configs/cifar10_line_paper.yaml
python -m atlas_one_step.cli build-atlas --config configs/cifar10_line_paper.yaml
python -m atlas_one_step.cli fit-surrogate --atlas outputs/cifar10_line_paper/atlas/atlas.parquet
python -m atlas_one_step.cli select-target \
  --atlas outputs/cifar10_line_paper/atlas/atlas.parquet \
  --surrogate outputs/cifar10_line_paper/atlas/surrogate.joblib \
  --output outputs/cifar10_line_paper/atlas/selected_target.json \
  --config configs/cifar10_train_paper.yaml

# Medium-scale validation on ImageNet-64 / 64x64 (requires imagefolder data)
python -m atlas_one_step.cli train --config configs/imagenet64_train_paper.yaml --mode coupled
python -m atlas_one_step.cli train --config configs/imagenet64_train_paper.yaml --mode atlas_guided \
  --selected-target outputs/cifar10_line_paper/atlas/selected_target.json

# Export plot-ready tables and compact summaries
python analysis/paper_results.py \
  --summary-root outputs \
  --atlas outputs/cifar10_line_paper/atlas/atlas.parquet \
  --outdir outputs/paper_results
