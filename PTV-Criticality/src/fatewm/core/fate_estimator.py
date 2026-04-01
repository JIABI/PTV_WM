import torch
import torch.nn as nn


class FateClassifier(nn.Module):
    """Predict a 4-way fate distribution from phi features.

    Outputs logits for 4 classes:
      0: Dissipate (gain < 1-eps_g)
      1: Transport (near-neutral gain, high transport)
      2: Amplify (gain > 1+eps_g)
      3: Large (large initial perturbation; optional emphasis)
    """

    def __init__(self, in_dim: int = -1, hidden: int = 64, n_classes: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.net(phi)


# Backward-compat alias used across the codebase.
FateEstimator = FateClassifier
