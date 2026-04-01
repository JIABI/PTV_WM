"""Environment adapter factory for PTV-Boundary."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .atari_adapter import AtariEnvAdapter
from .base import BaseEnvAdapter, EnvSpec, StepOutput
from .crafter_adapter import CrafterEnvAdapter
from .dmc_adapter import DMControlEnvAdapter
from .dummy_env import DummyEnv
from .myosuite_adapter import MyoSuiteEnvAdapter
from .procgen_adapter import ProcgenEnvAdapter


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def make_env(cfg: Any) -> BaseEnvAdapter:
    """Instantiate an environment adapter from config-like input.

    Parameters
    ----------
    cfg:
        Hydra DictConfig, mapping, or lightweight object with ``name`` and
        adapter-specific fields.
    """
    name = str(_cfg_get(cfg, "name", "dummy"))
    if name == "dummy":
        return DummyEnv(
            obs_dim=int(_cfg_get(cfg, "obs_dim", 16)),
            action_dim=int(_cfg_get(cfg, "action_dim", 6)),
            obs_type=str(_cfg_get(cfg, "obs_type", "proprio")),
            action_type=str(_cfg_get(cfg, "action_type", "discrete")),
            horizon=int(_cfg_get(cfg, "horizon", 16)),
        )
    if name == "atari100k":
        return AtariEnvAdapter(
            env_id=str(_cfg_get(cfg, "env_id", "ALE/Pong-v5")),
            obs_size=int(_cfg_get(cfg, "obs_size", 84)),
            frame_skip=int(_cfg_get(cfg, "frame_skip", 4)),
            noop_max=int(_cfg_get(cfg, "noop_max", 30)),
            frame_stack=int(_cfg_get(cfg, "frame_stack", 4)),
            grayscale_obs=bool(_cfg_get(cfg, "grayscale_obs", False)),
            full_action_space=bool(_cfg_get(cfg, "full_action_space", False)),
            episodic_life=bool(_cfg_get(cfg, "episodic_life", False)),
            repeat_action_probability=float(_cfg_get(cfg, "repeat_action_probability", 0.0)),
            max_episode_steps=int(_cfg_get(cfg, "max_episode_steps", 108000)),
            render_mode=_cfg_get(cfg, "render_mode", None),
        )
    if name in {"dmc_vision", "dmc_proprio"}:
        return DMControlEnvAdapter(
            domain_name=str(_cfg_get(cfg, "domain_name", "cartpole")),
            task_name=str(_cfg_get(cfg, "task_name", "swingup")),
            from_pixels=bool(_cfg_get(cfg, "from_pixels", name == "dmc_vision")),
            image_size=int(_cfg_get(cfg, "image_size", 84)),
            camera_id=int(_cfg_get(cfg, "camera_id", 0)),
            action_repeat=int(_cfg_get(cfg, "action_repeat", 1)),
            max_episode_steps=int(_cfg_get(cfg, "max_episode_steps", 1000)),
        )
    if name == "crafter":
        return CrafterEnvAdapter(
            env_id=str(_cfg_get(cfg, "env_id", "CrafterReward-v1")),
            max_episode_steps=int(_cfg_get(cfg, "max_episode_steps", 10000)),
            seed=int(_cfg_get(cfg, "seed", 0)),
        )
    if name == "procgen":
        return ProcgenEnvAdapter(
            env_id=str(_cfg_get(cfg, "env_id", "procgen:procgen-coinrun-v0")),
            start_level=int(_cfg_get(cfg, "start_level", 0)),
            num_levels=int(_cfg_get(cfg, "num_levels", 200)),
            distribution_mode=str(_cfg_get(cfg, "distribution_mode", "easy")),
            max_episode_steps=int(_cfg_get(cfg, "max_episode_steps", 1000)),
            render_mode=_cfg_get(cfg, "render_mode", None),
        )
    if name == "highdim_continuous":
        return MyoSuiteEnvAdapter(
            env_id=str(_cfg_get(cfg, "env_id", "myoFingerReachFixed-v0")),
            max_episode_steps=int(_cfg_get(cfg, "max_episode_steps", 1000)),
        )
    raise KeyError(f"Unknown environment adapter name: {name}")


__all__ = [
    "BaseEnvAdapter",
    "EnvSpec",
    "StepOutput",
    "DummyEnv",
    "AtariEnvAdapter",
    "DMControlEnvAdapter",
    "CrafterEnvAdapter",
    "ProcgenEnvAdapter",
    "MyoSuiteEnvAdapter",
    "make_env",
]
