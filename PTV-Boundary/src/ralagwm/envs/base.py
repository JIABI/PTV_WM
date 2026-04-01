from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class EnvSpec:
    name: str
    obs_type: str
    action_type: str
    observation_shape: tuple[int, ...]
    action_dim: int
    max_episode_steps: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self.observation_shape

    @property
    def num_actions(self) -> int:
        return self.action_dim


@dataclass(slots=True)
class StepOutput:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    @property
    def done(self) -> bool:
        return bool(self.terminated or self.truncated)

    def __iter__(self):
        yield self.observation
        yield self.reward
        yield self.done
        yield self.info

    def __len__(self) -> int:
        return 4


class BaseEnvAdapter(ABC):
    def __init__(self) -> None:
        self._spec: EnvSpec | None = None

    def __getattr__(self, item: str):
        """Delegate unknown attributes to the wrapped env when available.

        This lets adapter instances expose Gym-like attributes such as
        ``action_space`` and ``observation_space`` without each adapter having
        to re-declare forwarding properties.
        """
        wrapped = self.__dict__.get('env', None)
        if wrapped is not None:
            return getattr(wrapped, item)
        raise AttributeError(f"{self.__class__.__name__!s} has no attribute {item!r}")

    @property
    def spec(self) -> EnvSpec:
        if self._spec is None:
            raise RuntimeError('Environment adapter spec has not been initialized.')
        return self._spec

    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        ...

    @abstractmethod
    def step(self, action: Any) -> StepOutput:
        ...

    @abstractmethod
    def sample_random_action(self) -> Any:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
