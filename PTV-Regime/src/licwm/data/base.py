"""Dataset base and fallback synthetic generator."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import torch
from torch.utils.data import Dataset, random_split
from .batch_types import MultiAgentBatch

class BaseSequenceDataset(Dataset):
    """Base sequence dataset contract for all adapters."""
    def __init__(self, seq_len: int, pred_len: int):
        self.seq_len = seq_len
        self.pred_len = pred_len

@dataclass
class SplitSpec:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

class TinySyntheticFallbackDataset(BaseSequenceDataset):
    """Deterministic synthetic trajectories used when external data are unavailable."""
    def __init__(self, num_samples: int, seq_len: int, pred_len: int, n_agents: int, obs_dim: int, act_dim: int, event_dim: int, seed: int = 0):
        super().__init__(seq_len, pred_len)
        rng = np.random.default_rng(seed)
        self.samples = []
        t = np.linspace(0, 1, seq_len + pred_len)
        for i in range(num_samples):
            n = int(max(2, n_agents + (i % 3) - 1))
            phase = rng.uniform(0, 2 * math.pi)
            traj = np.stack([np.sin(t + phase), np.cos(t + phase)], axis=-1)
            traj = np.tile(traj[:, None, :], (1, n, 1)) + rng.normal(scale=0.03, size=(seq_len + pred_len, n, 2))
            obs = np.pad(traj, ((0, 0), (0, 0), (0, max(0, obs_dim - 2))))
            actions = np.diff(obs[:, :, :2], axis=0, prepend=obs[:1, :, :2])
            actions = np.pad(actions, ((0, 0), (0, 0), (0, max(0, act_dim - 2))))
            events = np.zeros((seq_len, event_dim), dtype=np.float32)
            events[:, 0] = (np.arange(seq_len) > seq_len // 2).astype(np.float32)
            mask = np.ones((seq_len + pred_len, n), dtype=np.float32)
            self.samples.append(MultiAgentBatch(
                obs_hist=torch.tensor(obs[:seq_len], dtype=torch.float32),
                action_hist=torch.tensor(actions[:seq_len], dtype=torch.float32),
                event_hist=torch.tensor(events, dtype=torch.float32),
                fut_state=torch.tensor(obs[seq_len:seq_len+pred_len], dtype=torch.float32),
                fut_obs=torch.tensor(obs[seq_len:seq_len+pred_len], dtype=torch.float32),
                mask=torch.tensor(mask[:seq_len], dtype=torch.float32),
                meta={"fallback": True},
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def split_dataset(ds: Dataset, spec: SplitSpec, seed: int = 0):
    n = len(ds)
    n_train = int(n * spec.train)
    n_val = int(n * spec.val)
    n_test = n - n_train - n_val
    return random_split(ds, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(seed))
