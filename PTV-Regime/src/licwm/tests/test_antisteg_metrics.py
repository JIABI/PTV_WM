import torch
from licwm.metrics.antisteg import tv_law, hf_law

def test_antisteg_nonconstant():
    law = torch.linspace(0, 1, steps=20).view(1,20,1).repeat(2,1,3)
    assert tv_law(law) > 0
    assert hf_law(law) >= 0
