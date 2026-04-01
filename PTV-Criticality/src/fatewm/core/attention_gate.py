import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    """AttentionGate: phi -> logits_over_deltas."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, phi: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        logits = self.net(phi) / max(float(temperature), 1e-6)
        return torch.softmax(logits, dim=-1)
