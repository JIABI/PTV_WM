from fatewm.algos.minimal.adapter import MinimalAdapter
from fatewm.algos.dreamerv3.adapter import DreamerV3Adapter
from fatewm.algos.tdmpc2.adapter import TDMPC2Adapter
from fatewm.algos.drqv2.adapter import DrQv2Adapter


def make_algo(cfg, obs_shape, action_dim: int, is_discrete: bool, latent_dim: int, hidden_dim: int):
    name = str(cfg.name)
    registry = {
        "minimal": MinimalAdapter,
        "dreamerv3": DreamerV3Adapter,
        "tdmpc2": TDMPC2Adapter,
        "drqv2": DrQv2Adapter,
    }
    if name not in registry:
        raise ValueError(f"Unknown algo: {name}")
    return registry[name](obs_shape, action_dim, is_discrete, latent_dim, hidden_dim)
