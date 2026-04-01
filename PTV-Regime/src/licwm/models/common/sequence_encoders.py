"""History encoders for H_t=(o<=t,a<t,e<=t)."""
import torch
from torch import nn

class HistoryEncoder(nn.Module):
    """Encode observation/action/event histories into climate posterior params."""
    def __init__(self, obs_dim: int, act_dim: int, event_dim: int, hidden_dim: int, c_dim: int, stochastic: bool = False):
        super().__init__()
        self.stochastic = stochastic
        in_dim = obs_dim + act_dim + event_dim
        self.rnn = nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.mu = nn.Linear(hidden_dim, c_dim)
        self.logvar = nn.Linear(hidden_dim, c_dim)

    def forward(self, obs_hist, action_hist=None, event_hist=None):
        b, t, n, d = obs_hist.shape
        obs_sys = obs_hist.mean(dim=2)
        act = action_hist.mean(dim=2) if action_hist is not None and action_hist.numel() > 0 else obs_sys.new_zeros((b, t, 0))
        evt = event_hist if event_hist is not None else obs_sys.new_zeros((b, t, 0))
        seq = torch.cat([obs_sys, act, evt], dim=-1)
        out, _ = self.rnn(seq)
        h = out[:, -1]
        mu, logvar = self.mu(h), self.logvar(h).clamp(-6, 6)
        if self.stochastic:
            eps = torch.randn_like(mu)
            c = mu + torch.exp(0.5 * logvar) * eps
        else:
            c = mu
        return {"c": c, "mu": mu, "logvar": logvar}
