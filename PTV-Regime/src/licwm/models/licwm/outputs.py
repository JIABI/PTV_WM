from dataclasses import dataclass
import torch

@dataclass
class LICWMOutputs:
    pred_states: torch.Tensor
    pred_obs: torch.Tensor
    fast_hidden: torch.Tensor
    climate_states: torch.Tensor
    jump_gates: torch.Tensor
    law_states: dict
    aux: dict
