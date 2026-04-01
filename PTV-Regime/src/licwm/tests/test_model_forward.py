import torch
from omegaconf import OmegaConf
from licwm.models.builders import build_model

def test_model_forward():
    cfg = OmegaConf.create({"domain":{"obs_dim":4,"action_dim":2,"event_dim":5},"model":{"name":"lic_wm","h_fast":16,"c_dim":8,"num_prototypes":3,"omega_max":0.2,"use_event_token":True,"transition_mode":"full","stochastic_climate":False,"enable_residual_channel":False,"residual_eps":0.05,"fast_mode":"gru"}})
    m = build_model(cfg)
    out = m(torch.randn(1,7,4,4), torch.randn(1,7,4,2), torch.randn(1,7,5), horizon=5)
    assert "beta" in out.law_states
