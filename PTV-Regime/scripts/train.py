import os
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
import hydra
from omegaconf import DictConfig
from licwm.training.engine import run_training

os.chdir(PROJECT_ROOT)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    run_training(cfg)

if __name__ == "__main__":
    main()
