"""Atari environment adapter built on Gymnasium Atari."""
from __future__ import annotations

from typing import Any

import numpy as np

from ralagwm.utils.import_guard import require_dependency

from .base import BaseEnvAdapter, EnvSpec, StepOutput
from .wrappers import preprocess_observation, safe_reset_gym_like, safe_step_gym_like


class AtariEnvAdapter(BaseEnvAdapter):
    """Adapter for ALE/Gymnasium Atari environments.

    The implementation follows the standard Gymnasium Atari stack using
    ``ALE/<game>-v5`` with ``AtariPreprocessing`` and optional
    ``FrameStackObservation``. This mirrors the commonly used preprocessing
    pipeline for Atari control benchmarks.
    """

    def __init__(
        self,
        env_id: str = "ALE/Pong-v5",
        obs_size: int = 84,
        frame_skip: int = 4,
        noop_max: int = 30,
        frame_stack: int = 4,
        grayscale_obs: bool = False,
        full_action_space: bool = False,
        episodic_life: bool = False,
        repeat_action_probability: float = 0.0,
        max_episode_steps: int = 108000,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        gym = require_dependency("gymnasium", "pip install gymnasium[atari,accept-rom-license] ale-py")
        try:
            import ale_py
            gym.register_envs(ale_py)
        except Exception:
            pass
        wrappers = require_dependency("gymnasium.wrappers", "Gymnasium Atari wrappers are required.")
        env = gym.make(
            env_id,
            render_mode=render_mode,
            frameskip=1,
            repeat_action_probability=float(repeat_action_probability),
            full_action_space=bool(full_action_space),
        )
        env = wrappers.AtariPreprocessing(
            env,
            noop_max=int(noop_max),
            frame_skip=int(frame_skip),
            screen_size=int(obs_size),
            terminal_on_life_loss=bool(episodic_life),
            grayscale_obs=bool(grayscale_obs),
            scale_obs=False,
        )
        if int(frame_stack) > 1:
            env = wrappers.FrameStackObservation(env, stack_size=int(frame_stack))
        self.env = env
        sample_obs, _ = safe_reset_gym_like(self.env, seed=0)
        obs = preprocess_observation(sample_obs, "image")
        self._spec = EnvSpec(
            name="atari100k",
            obs_type="image",
            action_type="discrete",
            observation_shape=tuple(obs.shape),
            action_dim=int(self.env.action_space.n),
            max_episode_steps=int(max_episode_steps),
            metadata={"env_id": env_id, "frame_stack": int(frame_stack)},
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
            frame = self.env.render()
            if frame is None:
                return None
            return preprocess_observation(frame, "image")
        except Exception:
            return None

    def close(self) -> None:
        self.env.close()
