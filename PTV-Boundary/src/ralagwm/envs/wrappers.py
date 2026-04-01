"""Environment compatibility utilities and lightweight wrappers."""
from __future__ import annotations

from inspect import signature
from typing import Any

import numpy as np
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_
import torch

from .base import StepOutput


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors/lists/lazy frames to ``np.ndarray``."""
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def flatten_obs_dict(obs: dict[str, Any]) -> np.ndarray:
    """Flatten a dictionary observation into a 1D float32 vector.

    Keys are processed in sorted order for determinism.
    """
    pieces: list[np.ndarray] = []
    for key in sorted(obs.keys()):
        arr = _to_numpy(obs[key]).astype(np.float32)
        pieces.append(arr.reshape(-1))
    if not pieces:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(pieces, axis=0)


def preprocess_image(obs: Any, channel_first: bool = True, normalize: bool = True) -> np.ndarray:
    """Convert image-like observation to float32 array.

    Handles lazy frames and images with optional leading batch dimension of size
    one, which is common for dm_control pixels wrappers.
    """
    arr = _to_numpy(obs)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if normalize:
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.float32)
    if channel_first and arr.ndim == 3:
        arr = np.transpose(arr, (2, 0, 1))
    return arr


def preprocess_observation(obs: Any, obs_type: str) -> np.ndarray:
    """Normalize observations to the repository-wide ndarray convention."""
    if isinstance(obs, dict):
        if obs_type == "image" and "pixels" in obs:
            return preprocess_image(obs["pixels"])
        return flatten_obs_dict(obs)
    arr = _to_numpy(obs)
    if obs_type == "image":
        return preprocess_image(arr)
    return arr.astype(np.float32).reshape(-1)


def safe_reset_gym_like(env: Any, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
    """Reset a Gymnasium or legacy Gym environment."""
    try:
        if seed is not None and "seed" in signature(env.reset).parameters:
            out = env.reset(seed=seed)
        else:
            out = env.reset()
    except TypeError:
        out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return obs, dict(info)
    return out, {}


def safe_step_gym_like(env: Any, action: Any, obs_type: str) -> StepOutput:
    """Step either a Gymnasium or legacy Gym environment."""
    out = env.step(action)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        terminated, truncated = bool(done), False
    else:  # pragma: no cover - defensive
        raise RuntimeError(f"Unsupported step return format: {type(out)}")
    return StepOutput(
        observation=preprocess_observation(obs, obs_type),
        reward=float(reward or 0.0),
        terminated=bool(terminated),
        truncated=bool(truncated),
        info=dict(info),
    )


class ActionRepeatWrapper:
    """Simple action repeat wrapper for adapter-internal usage."""

    def __init__(self, env: Any, repeat: int) -> None:
        self.env = env
        self.repeat = max(1, int(repeat))

    def step(self, action: Any):
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}
        obs = None
        for _ in range(self.repeat):
            out = self.env.step(action)
            if len(out) == 5:
                obs, reward, terminated, truncated, info = out
            else:
                obs, reward, done, info = out
                terminated, truncated = bool(done), False
            total_reward += float(reward or 0.0)
            if terminated or truncated:
                break
        if len(out) == 5:
            return obs, total_reward, terminated, truncated, info
        return obs, total_reward, bool(terminated or truncated), info

    def reset(self, *args: Any, **kwargs: Any):
        return self.env.reset(*args, **kwargs)

    def __getattr__(self, item: str) -> Any:
        return getattr(self.env, item)
