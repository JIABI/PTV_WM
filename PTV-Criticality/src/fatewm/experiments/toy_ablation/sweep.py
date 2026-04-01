import itertools, subprocess, sys
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../../../configs", config_name="default")
def main(cfg: DictConfig):
    sweep_cfg = hydra.compose(config_name="sweep/toy_grid")
    grid = sweep_cfg["grid"]
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    combos = list(itertools.product(*values))
    print(f"Toy sweep combos: {len(combos)}")

    for i, combo in enumerate(combos):
        overrides = dict(zip(keys, combo))
        if "env" not in overrides:
            overrides["env"] = "toy"
        if "method" not in overrides:
            overrides["method"] = "fatewm"
        if "algo" not in overrides:
            overrides["algo"] = cfg.algo.name

        cmd = [sys.executable, "-m", "fatewm.experiments.toy_ablation.train"]
        cmd += [f"{k}={v}" for k, v in overrides.items()]
        print(f"[{i+1}/{len(combos)}] {' '.join(cmd)}")
        subprocess.run(cmd, check=False)

if __name__ == "__main__":
    main()
