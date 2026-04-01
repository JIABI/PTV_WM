#!/usr/bin/env bash
set -euo pipefail

python analysis/paper_results.py \
  --summary-root outputs \
  --atlas outputs/cifar10_line_paper/atlas/atlas.parquet \
  --outdir outputs/paper_results
