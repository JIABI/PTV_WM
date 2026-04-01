import torch
from torch import nn
from licwm.models.licwm.outputs import LICWMOutputs

class TransformerWorldModel(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, h_dim: int = 64):
        super().__init__()
        self.proj = nn.Linear(obs_dim + action_dim, h_dim)
        enc = nn.TransformerEncoderLayer(d_model=h_dim, nhead=4, batch_first=True)
        self.tr = nn.TransformerEncoder(enc, num_layers=2)
        self.dec = nn.Linear(h_dim, obs_dim)

    def forward(self, obs_hist, action_hist=None, **_):
        b,t,n,d = obs_hist.shape
        a = action_hist if action_hist is not None and action_hist.numel()>0 else obs_hist.new_zeros((b,t,n,0))
        seq = torch.cat([obs_hist, a], dim=-1).reshape(b*n, t, -1)
        h = self.tr(self.proj(seq))
        pred = self.dec(h[:, 1:]).reshape(b, n, t-1, d).transpose(1,2)
        z = obs_hist.new_zeros((b, t-1, 1))
        return LICWMOutputs(pred, pred, h.reshape(b,n,t,-1).transpose(1,2)[:,1:], z, z, {"rho":z,"beta":z,"tau":z}, {})
