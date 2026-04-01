from .toy import ToyEnv
from .dmc import DMCEnv
from .atari import AtariEnv
from .procgen import ProcgenEnv

def make_env(cfg):
    if cfg.name == "toy":
        return ToyEnv(**{k:v for k,v in cfg.items() if k != "name"})
    if cfg.name == "dmc":
        return DMCEnv(cfg.task, cfg.from_pixels, cfg.frame_skip, cfg.episode_len)
    if cfg.name == "atari100k":
        return AtariEnv(cfg.game, cfg.frame_skip, cfg.sticky_actions, cfg.episode_len)
    if cfg.name == "procgen":
        return ProcgenEnv(cfg.env_name, cfg.num_levels, cfg.start_level, cfg.episode_len, cfg.get("frame_stack", 1))
    raise ValueError(f"Unknown env: {cfg.name}")
