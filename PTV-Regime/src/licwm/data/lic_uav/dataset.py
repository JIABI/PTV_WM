import torch
from torch.utils.data import Dataset
from .simulator import UAVSimulator
from ..batch_types import MultiAgentBatch

class LICUAVDataset(Dataset):
    def __init__(self, split: str, scenario: str, n_samples: int, n_agents: int, seq_len: int, pred_len: int):
        sim = UAVSimulator(n_agents=n_agents, scenario=scenario)
        self.items = []
        horizon = seq_len + pred_len
        for _ in range(n_samples):
            obs, events = sim.rollout(horizon)
            hist = torch.tensor(obs[:seq_len], dtype=torch.float32)
            fut = torch.tensor(obs[seq_len:], dtype=torch.float32)
            act = torch.diff(hist[..., :2], dim=0, prepend=hist[:1, ..., :2])
            self.items.append(MultiAgentBatch(obs_hist=hist, action_hist=act, event_hist=torch.tensor(events[:seq_len], dtype=torch.float32), fut_state=fut, fut_obs=fut, mask=torch.ones(seq_len, n_agents), meta={"split": split, "scenario": scenario}))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]
