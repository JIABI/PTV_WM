import torch
from torch.utils.data import Dataset
from .preprocess import load_or_mock
from .events import build_crowd_events
from ..batch_types import MultiAgentBatch

class ETHUCYDataset(Dataset):
    def __init__(self, split: str, data_root: str, seq_len: int, pred_len: int, use_velocity_proxy: bool = True):
        traj = load_or_mock(data_root, split)
        windows = traj.shape[0] - (seq_len + pred_len)
        self.items = []
        for i in range(max(1, windows)):
            hist = torch.tensor(traj[i:i+seq_len], dtype=torch.float32)
            fut = torch.tensor(traj[i+seq_len:i+seq_len+pred_len], dtype=torch.float32)
            acts = torch.diff(hist, dim=0, prepend=hist[:1]) if use_velocity_proxy else torch.zeros_like(hist[..., :0])
            self.items.append(MultiAgentBatch(obs_hist=hist, action_hist=acts, event_hist=build_crowd_events(hist), fut_state=fut, fut_obs=fut, mask=torch.ones(seq_len, hist.size(1)), meta={"split": split, "source": "eth_ucy"}))
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]
