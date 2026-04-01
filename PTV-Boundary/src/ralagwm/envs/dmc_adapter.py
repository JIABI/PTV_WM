"""DeepMind Control Suite adapter."""
from __future__ import annotations

from typing import Any

import numpy as np

from ralagwm.utils.import_guard import require_dependency

from .base import BaseEnvAdapter, EnvSpec, StepOutput
from .wrappers import flatten_obs_dict, preprocess_observation


class DMControlEnvAdapter(BaseEnvAdapter):
    """Adapter for dm_control tasks.

    Supports both proprioceptive observations and pixel observations via the
    official dm_control pixels wrapper.
    """

    def __init__(
        self,
        domain_name: str = "cartpole",
        task_name: str = "swingup",
        from_pixels: bool = False,
        image_size: int = 84,
        camera_id: int = 0,
        action_repeat: int = 1,
        max_episode_steps: int = 1000,
    ) -> None:
        super().__init__()
        suite = require_dependency("dm_control.suite", "pip install dm-control")
        task_kwargs = {"random": 0}
        env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs)
        self.from_pixels = bool(from_pixels)
        if self.from_pixels:
            pixels = require_dependency("dm_control.suite.wrappers.pixels", "dm_control pixels wrapper is required.")
            env = pixels.Wrapper(
                env,
                pixels_only=False,
                render_kwargs={"height": int(image_size), "width": int(image_size), "camera_id": int(camera_id)},
            )
        self.env = env
        self.action_repeat = max(1, int(action_repeat))
        obs, _ = self.reset(seed=0)
        action_spec = self.env.action_spec()
        self._minimum = np.asarray(action_spec.minimum, dtype=np.float32)
        self._maximum = np.asarray(action_spec.maximum, dtype=np.float32)
        self._spec = EnvSpec(
            name="dmc_vision" if self.from_pixels else "dmc_proprio",
            obs_type="image" if self.from_pixels else "proprio",
            action_type="continuous",
            observation_shape=tuple(obs.shape),
            action_dim=int(np.prod(action_spec.shape)),
            max_episode_steps=int(max_episode_steps),
            metadata={"domain": domain_name, "task": task_name, "from_pixels": self.from_pixels},
        )

    def _process_timestep_obs(self, time_step: Any) -> np.ndarray:
        obs = time_step.observation
        if self.from_pixels:
            if isinstance(obs, dict) and "pixels" in obs:
                return preprocess_observation(obs["pixels"], "image")
            return preprocess_observation(obs, "image")
        if isinstance(obs, dict):
            return flatten_obs_dict(obs)
        return np.asarray(obs, dtype=np.float32).reshape(-1)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            # dm_control seeds through task_kwargs during construction; keeping
            # adapter contract stable by rebuilding is intentionally avoided here.
            pass
        ts = self.env.reset()
        return self._process_timestep_obs(ts), {}

    def step(self, action: np.ndarray) -> StepOutput:
        action_arr = np.asarray(action, dtype=np.float32).reshape(self._minimum.shape)
        action_arr = np.clip(action_arr, self._minimum, self._maximum)
        total_reward = 0.0
        ts = None
        for _ in range(self.action_repeat):
            ts = self.env.step(action_arr)
            total_reward += float(ts.reward or 0.0)
            if ts.last():
                break
        assert ts is not None
        return StepOutput(
            observation=self._process_timestep_obs(ts),
            reward=total_reward,
            terminated=bool(ts.last()),
            truncated=False,
            info={"discount": float(ts.discount) if ts.discount is not None else 1.0},
        )

    def sample_random_action(self) -> np.ndarray:
        return np.random.uniform(self._minimum, self._maximum).astype(np.float32)

    def close(self) -> None:
        close_fn = getattr(self.env, "close", None)
        if callable(close_fn):
            close_fn()
