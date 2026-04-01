import torch
from ralagwm.typing import ChartState
from ralagwm.chart.generator import BICChartGenerator

def test_bic_chart_legal():
    cstate = ChartState(
        anchor_action=torch.tensor(0),
        metric_matrix=torch.eye(8),
        boundary_saliency=torch.ones(6),
        uncertainty=torch.zeros(6),
        action_coords=torch.randn(6, 8),
        metadata={},
    )
    generator = BICChartGenerator(chart_budget=4, feature_dim=8)
    chart = generator.generate(cstate, torch.randn(6), torch.rand(6) * 0.1)
    assert chart.selected_indices.shape[0] == 4


def test_bic_chart_continuous_anchor_vector():
    cstate = ChartState(
        anchor_action=torch.tensor([0.1, -0.2, 0.3, 0.0, 0.0, 0.0]),
        metric_matrix=torch.eye(6),
        boundary_saliency=torch.linspace(0.0, 1.0, 6),
        uncertainty=torch.zeros(6),
        action_coords=torch.stack([
            torch.tensor([0.1, -0.2, 0.3, 0.0, 0.0, 0.0]),
            torch.tensor([0.2, -0.2, 0.3, 0.0, 0.0, 0.0]),
            torch.tensor([0.1, -0.1, 0.3, 0.0, 0.0, 0.0]),
            torch.tensor([0.1, -0.2, 0.4, 0.0, 0.0, 0.0]),
            torch.tensor([-0.1, -0.2, 0.3, 0.0, 0.0, 0.0]),
            torch.tensor([0.1, -0.3, 0.3, 0.0, 0.0, 0.0]),
        ], dim=0),
        metadata={'action_type': 'continuous'},
    )
    generator = BICChartGenerator(chart_budget=4, feature_dim=7, pool_budget=6, mode='continuous')
    chart = generator.generate(cstate, torch.linspace(0.0, 1.0, 6), torch.rand(6) * 0.1)
    assert chart.selected_indices.shape[0] == 4
    assert int(chart.selected_indices[0].item()) == 0
