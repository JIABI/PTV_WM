"""Procgen environment adapter."""
from __future__ import annotations

import numpy as np

from ralagwm.utils.import_guard import require_dependency

from .base import BaseEnvAdapter, EnvSpec, StepOutput
from .wrappers import preprocess_observation, safe_reset_gym_like, safe_step_gym_like


class ProcgenEnvAdapter(BaseEnvAdapter):
    """Adapter for the official Procgen Gym environments."""

    def __init__(
        self,
        env_id: str = "procgen:procgen-coinrun-v0",
        start_level: int = 0,
        num_levels: int = 200,
        distribution_mode: str = "easy",
        max_episode_steps: int = 1000,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        require_dependency("procgen", "pip install procgen gym")
        gym = require_dependency("gym", "Procgen currently exposes a legacy Gym API.")
        env = gym.make(
            env_id,
            start_level=int(start_level),
            num_levels=int(num_levels),
            distribution_mode=distribution_mode,
            render_mode=render_mode,
        )
        self.env = env
        obs, _ = safe_reset_gym_like(self.env, seed=0)
        proc = preprocess_observation(obs, "image")
        self._spec = EnvSpec(
            name="procgen",
            obs_type="image",
            action_type="discrete",
            observation_shape=tuple(proc.shape),
            action_dim=int(self.env.action_space.n),
            max_episode_steps=int(max_episode_steps),
            metadata={
                "env_id": env_id,
                "start_level": int(start_level),
                "num_levels": int(num_levels),
                "distribution_mode": distribution_mode,
            },
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = safe_reset_gym_like(self.env, seed=seed)
        return preprocess_observation(obs, "image"), info

    def step(self, action: int | np.ndarray) -> StepOutput:
        return safe_step_gym_like(self.env, int(action), obs_type="image")

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
