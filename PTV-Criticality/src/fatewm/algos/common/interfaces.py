from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple
import torch
import torch.nn as nn

@dataclass
class AlgoOutputs:
    loss: torch.Tensor
    logs: Dict[str, float]

class AlgoAdapter(nn.Module):
    """Unifies different backbones under one interface for FATE-WM core.

    Required primitives:
      - encode(obs) -> latent z (any structured tensor/dict)
      - scores(z) -> action scores/logits/Q for margin/vulnerability
      - predict(z, a, delta) -> predicted latent at t+delta (optional for model-free)
      - update(batch) -> AlgoOutputs (training update step)
    """
    def __init__(self):
        super().__init__()

    @property
    def is_model_based(self) -> bool:
        return False

    def encode(self, obs: torch.Tensor):
        raise NotImplementedError

    def scores(self, z):
        raise NotImplementedError

    def predict(self, z, actions, delta: int):
        """Optional; only for model-based (Dreamer/TD-MPC2)."""
        raise NotImplementedError

    def update(self, batch, deltas, method_cfg, fate_estimator=None, att_gate=None) -> AlgoOutputs:
        raise NotImplementedError
