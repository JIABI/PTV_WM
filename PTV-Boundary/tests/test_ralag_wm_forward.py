import torch
from ralagwm.models.ralag_wm import RALAGWM

def test_model_forward():
    model = RALAGWM(obs_dim=16, action_dim=6)
    out = model(torch.randn(2, 16))
    assert out.latent.shape[-1] == 32


def test_model_forward_continuous_chart_generation():
    model = RALAGWM(obs_dim=24, action_dim=6, chart_mode='continuous', chart_budget=4, pool_budget=6)
    out = model(torch.randn(2, 24))
    assert out.pred_chart is not None
    assert out.pred_chart.actions.shape[1] == 4
    assert out.selected_action.shape[-1] == 6
