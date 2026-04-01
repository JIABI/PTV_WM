"""Minimal runnable environment adapter used for smoke tests."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .base import BaseEnvAdapter, EnvSpec, StepOutput
from .wrappers import preprocess_observation


class DummyEnv(BaseEnvAdapter):
    """A tiny environment supporting both discrete and continuous actions."""

    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 6,
        obs_type: str = "proprio",
        action_type: str = "discrete",
        horizon: int = 16,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.obs_type = obs_type
        self.action_type = action_type
        self.horizon = int(horizon)
        self._step_count = 0
        if self.obs_type == "image":
            obs_shape = (3, self.obs_dim, self.obs_dim)
        else:
            obs_shape = (self.obs_dim,)
        self._spec = EnvSpec(
            name="dummy",
            obs_type=self.obs_type,
            action_type=self.action_type,
            observation_shape=obs_shape,
            action_dim=self.action_dim,
            max_episode_steps=self.horizon,
            metadata={"dummy": True},
        )

    def _obs(self) -> np.ndarray:
        if self.obs_type == "image":
            obs = np.random.randint(0, 255, size=(self.obs_dim, self.obs_dim, 3), dtype=np.uint8)
            return preprocess_observation(obs, self.obs_type)
        return np.random.randn(self.obs_dim).astype(np.float32)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self._step_count = 0
        return self._obs(), {"seed": seed}

    def step(self, action: int | np.ndarray | torch.Tensor) -> StepOutput:
        self._step_count += 1
        reward = float(np.random.randn())
        terminated = self._step_count >= self.horizon
        return StepOutput(
            observation=self._obs(),
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={"dummy_action": np.asarray(action).tolist() if not isinstance(action, int) else int(action)},
        )

    def sample_random_action(self) -> int | np.ndarray:
        if self.action_type == "discrete":
            return int(np.random.randint(0, self.action_dim))
        return np.random.uniform(-1.0, 1.0, size=(self.action_dim,)).astype(np.float32)

    def close(self) -> None:
        return None
