import torch
from torch import nn
from .gru_wm import GRUWorldModel

class MoEWorldModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, experts: int = 3):
        super().__init__()
        self.experts = nn.ModuleList([GRUWorldModel(obs_dim, action_dim) for _ in range(experts)])
        self.gate = nn.Linear(obs_dim, experts)

    def forward(self, obs_hist, action_hist=None, event_hist=None, **kwargs):
        gates = torch.softmax(self.gate(obs_hist[:,0].mean(dim=1)), dim=-1)
        outs = [m(obs_hist, action_hist, event_hist, **kwargs) for m in self.experts]
        pred = sum(gates[:, i, None, None, None] * outs[i].pred_obs for i in range(len(outs)))
        ref = outs[0]
        ref.pred_obs = pred
        ref.pred_states = pred
        return ref
