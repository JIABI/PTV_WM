"""MyoSuite environment adapter."""
from __future__ import annotations

import numpy as np

from ralagwm.utils.import_guard import require_dependency

from .base import BaseEnvAdapter, EnvSpec, StepOutput
from .wrappers import preprocess_observation, safe_reset_gym_like, safe_step_gym_like


class MyoSuiteEnvAdapter(BaseEnvAdapter):
    """Adapter for MyoSuite musculoskeletal control tasks.

    MyoSuite registers legacy Gym environments after importing ``myosuite``.
    The adapter keeps the action/observation interface continuous and flattens
    proprioceptive observations.
    """

    def __init__(
        self,
        env_id: str = "myoFingerReachFixed-v0",
        max_episode_steps: int = 1000,
    ) -> None:
        super().__init__()
        require_dependency("myosuite", "pip install myosuite gym")
        gym = require_dependency("gym", "MyoSuite currently exposes a legacy Gym API.")
        env = gym.make(env_id)
        self.env = env
        obs, _ = safe_reset_gym_like(self.env, seed=0)
        proc = preprocess_observation(obs, "proprio")
        action_space = getattr(self.env, "action_space", None)
        if action_space is None:
            raise RuntimeError("MyoSuite environment did not expose an action_space.")
        self._minimum = np.asarray(action_space.low, dtype=np.float32)
        self._maximum = np.asarray(action_space.high, dtype=np.float32)
        self._spec = EnvSpec(
            name="highdim_continuous",
            obs_type="proprio",
            action_type="continuous",
            observation_shape=tuple(proc.shape),
            action_dim=int(np.prod(self._minimum.shape)),
            max_episode_steps=int(max_episode_steps),
            metadata={"env_id": env_id},
        )

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = safe_reset_gym_like(self.env, seed=seed)
        return preprocess_observation(obs, "proprio"), info

    def step(self, action: np.ndarray) -> StepOutput:
        action_arr = np.asarray(action, dtype=np.float32).reshape(self._minimum.shape)
        action_arr = np.clip(action_arr, self._minimum, self._maximum)
        return safe_step_gym_like(self.env, action_arr, obs_type="proprio")

    def sample_random_action(self) -> np.ndarray:
        return np.random.uniform(self._minimum, self._maximum).astype(np.float32)

    def close(self) -> None:
        self.env.close()
