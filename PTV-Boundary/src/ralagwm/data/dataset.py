"""Dataset wrappers for replay-driven training."""
from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset

from .replay import ReplayBuffer, Transition


class ReplayDataset(Dataset):
    """Thin dataset view over a replay buffer."""

    def __init__(self, replay: ReplayBuffer):
        self.replay = replay

    def __len__(self) -> int:
        return len(self.replay)

    def __getitem__(self, index: int) -> Transition:
        return self.replay.data[index]
