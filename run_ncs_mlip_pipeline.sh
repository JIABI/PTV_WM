#!/usr/bin/env bash
set -euo pipefail

python -m ptv_ncs_mlip.cli validate-archive --data-root data/ncs_mlip
python -m ptv_ncs_mlip.cli export-all --config configs/ncs_mlip_hero.yaml
