import torch
from torch.utils.data import Dataset
from .generator import generate_sequences
from ..batch_types import MultiAgentBatch

class LICBoidsDataset(Dataset):
    def __init__(self, split: str, mode: str, n_samples: int, n_agents: int, seq_len: int, pred_len: int, event_dim: int):
        self.items = []
        horizon = seq_len + pred_len
        for obs, events, law in generate_sequences(n_samples, n_agents, mode, horizon):
            self.items.append(MultiAgentBatch(
                obs_hist=torch.tensor(obs[:seq_len], dtype=torch.float32),
                action_hist=torch.tensor(obs[1:seq_len+1, :, 2:4] - obs[:seq_len, :, 2:4], dtype=torch.float32),
                event_hist=torch.tensor(events[:seq_len, :event_dim], dtype=torch.float32),
                fut_state=torch.tensor(obs[seq_len:], dtype=torch.float32),
                fut_obs=torch.tensor(obs[seq_len:], dtype=torch.float32),
                mask=torch.ones(seq_len, n_agents),
                meta={"split": split, "gt_law": torch.tensor(law[:seq_len])},
            ))

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]
