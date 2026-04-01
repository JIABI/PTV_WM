from __future__ import annotations
import torch
from torch import nn
from licwm.models.licwm.outputs import LICWMOutputs


class CFCWorldModel(nn.Module):
    """A light continuous-time-inspired baseline.

    Not a full CfC implementation, but a stronger non-LIC memory baseline than a plain GRU.
    """
    def __init__(self, obs_dim: int, action_dim: int, h_dim: int = 64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.h_dim = h_dim
        self.in_proj = nn.Linear(obs_dim + action_dim, h_dim)
        self.gate = nn.Linear(h_dim, h_dim)
        self.dec = nn.Linear(h_dim, obs_dim)

    def forward(self, obs_hist, action_hist=None, event_hist=None, horizon=None, teacher_forcing=True, **_):
        b, t, n, d = obs_hist.shape
        h = obs_hist.new_zeros((b * n, self.h_dim))
        x = obs_hist[:, 0]
        horizon = horizon or t - 1
        preds, hs = [], []
        for s in range(horizon):
            if teacher_forcing and s < t:
                x = obs_hist[:, s]
            a = action_hist[:, s] if action_hist is not None and action_hist.numel() > 0 and s < action_hist.shape[1] else x.new_zeros((b, n, self.action_dim))
            z = torch.tanh(self.in_proj(torch.cat([x, a], dim=-1).reshape(b * n, -1)))
            alpha = torch.sigmoid(self.gate(h))
            h = alpha * h + (1 - alpha) * z
            x = self.dec(h).reshape(b, n, d)
            preds.append(x); hs.append(h.reshape(b, n, -1))
        ztraj = obs_hist.new_zeros((b, horizon, 1))
        return LICWMOutputs(torch.stack(preds,1), torch.stack(preds,1), torch.stack(hs,1), ztraj, ztraj, {"rho":ztraj, "beta":ztraj, "tau":ztraj}, {})
