from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import hydra
from omegaconf import DictConfig

from ralagwm.training.paper_pipeline import train_audit_for_domain
from ralagwm.utils.logging import setup_logging


@hydra.main(version_base=None, config_path='../configs', config_name='base')
def main(cfg: DictConfig) -> None:
    setup_logging(str(getattr(cfg.logging, 'level', 'INFO')))
    train_audit_for_domain(cfg, run_name='single_train_audit')


if __name__ == '__main__':
    main()
