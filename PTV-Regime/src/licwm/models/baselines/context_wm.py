import torch
from torch import nn
from .gru_wm import GRUWorldModel

class ContextWorldModel(GRUWorldModel):
    def __init__(self, obs_dim: int, action_dim: int, context_dim: int = 8, h_dim: int = 64):
        super().__init__(obs_dim + context_dim, action_dim, h_dim)
        self.ctx = nn.Parameter(torch.zeros(context_dim))
        self.obs_dim = obs_dim

    def forward(self, obs_hist, *args, **kwargs):
        b,t,n,d = obs_hist.shape
        ctx = self.ctx[None,None,None,:].expand(b,t,n,-1)
        out = super().forward(torch.cat([obs_hist, ctx], dim=-1), *args, **kwargs)
        out.pred_obs = out.pred_obs[..., :self.obs_dim]
        out.pred_states = out.pred_obs
        return out
