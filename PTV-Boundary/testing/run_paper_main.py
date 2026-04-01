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

from ralagwm.evaluation.benchmark import run_benchmark
from ralagwm.evaluation.frontier import run_frontier
from ralagwm.evaluation.matched_fidelity import run_matched_fidelity
from ralagwm.evaluation.oracle_substitution import run_oracle_substitution
from ralagwm.evaluation.robustness import run_robustness
from ralagwm.utils.logging import setup_logging


@hydra.main(version_base=None, config_path='../configs', config_name='base')
def main(cfg: DictConfig) -> None:
    setup_logging(str(getattr(cfg.logging, 'level', 'INFO')))
    cfg.experiment.manifest_path = getattr(getattr(cfg, 'experiment', {}), 'manifest_path', '') or 'inputs/manifests/paper_main_domains.yaml'
    ckpt = str(getattr(cfg, 'checkpoint_path', 'outputs/checkpoints/ralag_wm.pt'))
    run_benchmark(cfg, checkpoint_path=ckpt)
    run_matched_fidelity(cfg, checkpoint_paths=[ckpt])
    run_oracle_substitution(cfg, checkpoint_path=ckpt)
    run_frontier(cfg, checkpoint_path=ckpt)
    run_robustness(cfg, checkpoint_path=ckpt)
    print('[paper-main] outputs written under outputs/metrics and outputs/figures')


if __name__ == '__main__':
    main()
