import torch
from ralagwm.typing import BICChart
from ralagwm.geometry.extractor import extract_ralag_geometry

def test_geometry_shapes():
    chart = BICChart(
        actions=torch.arange(4),
        coords=torch.randn(4, 8),
        edges=torch.tensor([[0, 1], [1, 2]]),
        weights=torch.ones(4),
        info_matrix=torch.eye(8),
        selected_indices=torch.arange(4),
    )
    g = extract_ralag_geometry(torch.randn(4), chart)
    assert g.centered_scores.shape[0] == 4
