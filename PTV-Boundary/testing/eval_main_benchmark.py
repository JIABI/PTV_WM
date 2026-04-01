from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import hydra
from omegaconf import DictConfig
from ralagwm.evaluation.benchmark import run_benchmark


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    run_benchmark(cfg, checkpoint_path=str(getattr(cfg, "checkpoint_path", "outputs/checkpoints/ralag_wm.pt")))


if __name__ == "__main__":
    main()
