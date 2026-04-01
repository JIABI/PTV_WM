import torch
from licwm.models.licwm.climate_transition import ClimateTransition

def test_jump_mode_shapes():
    c = torch.zeros(2, 8); q = torch.zeros(2, 16); e = torch.zeros(2, 5)
    full = ClimateTransition(8, 16, 5, mode="full")
    no_jump = ClimateTransition(8, 16, 5, mode="no_jump")
    assert full(c, q, e)["c_next"].shape == no_jump(c, q, e)["c_next"].shape
