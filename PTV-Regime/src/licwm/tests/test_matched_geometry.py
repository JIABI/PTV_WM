import torch
from licwm.metrics.matched_geometry import geometry_distance

def test_geometry_distance_zero_self():
    x = torch.zeros(4,2)
    assert geometry_distance(x, x) == 0.0
