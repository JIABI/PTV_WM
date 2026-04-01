"""Crafter environment adapter."""
from __future__ import annotations

from typing import Any
from inspect import signature

import numpy as np

# NumPy >= 2 removed the legacy alias used by old Gym checkers.
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from ralagwm.utils.import_guard import OptionalDependencyError, require_dependency

from .base import BaseEnvAdapter, EnvSpec, StepOutput
from .wrappers import preprocess_observation, safe_reset_gym_like, safe_step_gym_like


class CrafterEnvAdapter(BaseEnvAdapter):
    """Adapter for the official Crafter Gym interface.

    Crafter registers legacy Gym environments such as ``CrafterReward-v1``. The
    adapter uses the legacy gym API when available and normalizes it to the
    repository-wide adapter API.
    """

    def __init__(
        self,
        env_id: str = "CrafterReward-v1",
        max_episode_steps: int = 10000,
        seed: int = 0,
    ) -> None:
        super().__init__()
        require_dependency("crafter", "pip install crafter gym")
        gym = require_dependency("gym", "Crafter currently exposes a legacy Gym API.")
        make_kwargs = {}
        try:
            if "disable_env_checker" in signature(gym.make).parameters:
                make_kwargs["disable_env_checker"] = True
        except Exception:
            pass
        # Crafter currently exposes a legacy Gym-style API. Some Gym wrapper
        # stacks (notably TimeLimit in older gym versions) now expect a
        # 5-tuple step result and fail before our generic compatibility layer
        # can run. To keep the adapter robust across old/new gym releases, we
        # bypass the outer wrapper chain and talk to the unwrapped Crafter env
        # directly, while enforcing the episode limit inside the adapter.
        env = gym.make(env_id, **make_kwargs)
        self.env = getattr(env, "unwrapped", env)
        self._elapsed_steps = 0
        self._max_episode_steps = int(max_episode_steps)
        obs, _ = safe_reset_gym_like(self.env, seed=seed)
        proc = preprocess_observation(obs, "image")
        self._spec = EnvSpec(
            name="crafter",
            obs_type="image",
            action_type="discrete",
            observation_shape=tuple(proc.shape),
            action_dim=int(self.env.action_space.n),
            max_episode_steps=int(max_episode_steps),
            metadata={"env_id": env_id},
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self._elapsed_steps = 0
        obs, info = safe_reset_gym_like(self.env, seed=seed)
        return preprocess_observation(obs, "image"), info

    def step(self, action: int | np.ndarray) -> StepOutput:
        self._elapsed_steps += 1
        step_out = safe_step_gym_like(self.env, int(action), obs_type="image")
        # Legacy Crafter envs usually return a 4-tuple (obs, reward, done,
        # info) with no separate truncation bit. We enforce the adapter-level
        # max episode length here so downstream code continues to see a stable
        # Gymnasium-style terminated/truncated split.
        if not step_out.done and self._elapsed_steps >= self._max_episode_steps:
            return StepOutput(
                observation=step_out.observation,
                reward=step_out.reward,
                terminated=False,
                truncated=True,
                info=dict(step_out.info),
            )
        return step_out

    def sample_random_action(self) -> int:
        return int(self.env.action_space.sample())

    def render(self) -> np.ndarray | None:
        try:
            frame = self.env.render(mode="rgb_array")
        except TypeError:
            try:
                frame = self.env.render()
            except Exception:
                return None
        except Exception:
            return None
        if frame is None:
            return None
        return preprocess_observation(frame, "image")

    def close(self) -> None:
        self.env.close()
