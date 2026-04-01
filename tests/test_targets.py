import torch

from atlas_one_step.corruption import DiffusionLikeCorruption
from atlas_one_step.targets import TargetSpec, construct_target, reconstruct_x0_from_target


def test_target_reconstruction_roundtrip():
    x0 = torch.randn(8, 3, 16, 16)
    corr = DiffusionLikeCorruption()
    t = torch.full((8,), 0.4)
    xt, eps = corr.sample_xt(x0, t)
    state = corr.primitives(x0, xt, eps, t)
    state['t_scalar'] = t
    spec = TargetSpec('line_x0_r', {'alpha': 0.7})
    y = construct_target(spec, state)
    x0_hat = reconstruct_x0_from_target(spec, y, state)
    assert torch.allclose(x0, x0_hat, atol=1e-4, rtol=1e-4)
