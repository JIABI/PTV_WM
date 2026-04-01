import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from fatewm.runners.eval_loop import evaluate
from fatewm.experiments.atari100k.train import build_components
from fatewm.envs import make_env

@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = torch.device(cfg.device)
    components = build_components(cfg, device)
    env = make_env(cfg.env)
    stats = evaluate(cfg, components, n_episodes=int(cfg.eval.episodes), env=env)
    print(stats)

if __name__ == "__main__":
    main()
