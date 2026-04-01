"""Climate encoder q_phi(c_t|H_t)."""
from torch import nn
from ..common.sequence_encoders import HistoryEncoder

class ClimateEncoder(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, event_dim: int, hidden_dim: int, c_dim: int, stochastic: bool = False):
        super().__init__()
        self.encoder = HistoryEncoder(obs_dim, action_dim, event_dim, hidden_dim, c_dim, stochastic)

    def forward(self, obs_hist, action_hist=None, event_hist=None):
        return self.encoder(obs_hist=obs_hist, action_hist=action_hist, event_hist=event_hist)
