"""Replay buffer and transition utilities for environment-driven training."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any
import json
import random

import numpy as np
import torch

from ralagwm.data.batch import Transition


class ReplayBuffer:
    """Simple FIFO replay buffer with random mini-batch sampling."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = int(capacity)
        self.data: list[Transition] = []
        self._pos = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, item: Transition) -> None:
        if len(self.data) < self.capacity:
            self.data.append(item)
        else:
            self.data[self._pos] = item
        self._pos = (self._pos + 1) % self.capacity

    def extend(self, items: list[Transition]) -> None:
        for item in items:
            self.add(item)

    def sample(self, batch_size: int) -> list[Transition]:
        if not self.data:
            raise RuntimeError('ReplayBuffer is empty.')
        batch_size = min(int(batch_size), len(self.data))
        return random.sample(self.data, batch_size)

    def sample_tensors(self, batch_size: int, device: torch.device | str = 'cpu') -> dict[str, torch.Tensor]:
        batch = self.sample(batch_size)
        obs = torch.stack([torch.as_tensor(t.obs, dtype=torch.float32) for t in batch]).to(device)
        next_obs = torch.stack([torch.as_tensor(t.next_obs, dtype=torch.float32) for t in batch]).to(device)
        actions = torch.stack([
            torch.as_tensor(t.action, dtype=torch.float32).reshape(-1) for t in batch
        ]).to(device)
        rewards = torch.tensor([float(t.reward) for t in batch], dtype=torch.float32, device=device)
        dones = torch.tensor([float(t.done) for t in batch], dtype=torch.float32, device=device)
        return {
            'obs': obs,
            'next_obs': next_obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'transitions': batch,
        }

    def save_jsonl(self, path: str | Path, limit: int | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = self.data if limit is None else self.data[: int(limit)]
        with path.open('w') as f:
            for item in rows:
                payload = asdict(item)
                payload['obs'] = np.asarray(payload['obs']).tolist()
                payload['next_obs'] = np.asarray(payload['next_obs']).tolist()
                payload['action'] = np.asarray(payload['action']).tolist() if not isinstance(payload['action'], (int, float)) else payload['action']
                f.write(json.dumps(payload) + '\n')
