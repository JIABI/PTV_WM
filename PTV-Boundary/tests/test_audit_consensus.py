import torch
from ralagwm.audit.consensus import trimmed_mean_consensus, variance_disagreement

def test_consensus_shapes():
    raw = torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    cons = trimmed_mean_consensus(raw, 0.25)
    dis = variance_disagreement(raw)
    assert cons.shape == (2,)
    assert dis.shape == (2,)
