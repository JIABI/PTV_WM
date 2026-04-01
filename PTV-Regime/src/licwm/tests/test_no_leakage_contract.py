from omegaconf import OmegaConf
from licwm.models.builders import build_model

def test_residual_default_off():
    cfg = OmegaConf.create({"domain":{"obs_dim":4,"action_dim":2,"event_dim":5},"model":{"name":"lic_wm","h_fast":16,"c_dim":8,"num_prototypes":3,"omega_max":0.2,"use_event_token":True,"transition_mode":"full","stochastic_climate":False,"enable_residual_channel":False,"residual_eps":0.05,"fast_mode":"gru"}})
    model = build_model(cfg)
    assert model.residual is None
