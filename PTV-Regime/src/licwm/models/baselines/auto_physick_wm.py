import torch
from torch import nn
from .gru_wm import GRUWorldModel

class AutoPhysicKWorldModel(GRUWorldModel):
    """No-climate structured fast-law baseline."""
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__(obs_dim, action_dim)
        self.law_gain = nn.Parameter(torch.tensor(0.1))
