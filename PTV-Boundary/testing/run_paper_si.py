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
from ralagwm.evaluation.matched_fidelity import run_matched_fidelity
from ralagwm.evaluation.robustness import run_robustness
from ralagwm.utils.io import dump_json
from ralagwm.utils.logging import setup_logging


@hydra.main(version_base=None, config_path='../configs', config_name='base')
def main(cfg: DictConfig) -> None:
    setup_logging(str(getattr(cfg.logging, 'level', 'INFO')))
    cfg.experiment.manifest_path = getattr(getattr(cfg, 'experiment', {}), 'manifest_path', '') or 'inputs/manifests/paper_si_domains.yaml'
    ckpt = str(getattr(cfg, 'checkpoint_path', 'outputs/checkpoints/ralag_wm.pt'))
    out = {
        'benchmark': run_benchmark(cfg, checkpoint_path=ckpt),
        'matched_fidelity': run_matched_fidelity(cfg, checkpoint_paths=[ckpt]),
        'robustness': run_robustness(cfg, checkpoint_path=ckpt),
    }
    dump_json('outputs/metrics/paper_si_summary.json', out)
    print('[paper-si] outputs written under outputs/metrics')


if __name__ == '__main__':
    main()
