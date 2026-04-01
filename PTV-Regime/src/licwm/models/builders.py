from licwm.models.licwm.model import LICWorldModel
from licwm.models.baselines.gru_wm import GRUWorldModel
from licwm.models.baselines.context_wm import ContextWorldModel
from licwm.models.baselines.transformer_wm import TransformerWorldModel
from licwm.models.baselines.auto_physick_wm import AutoPhysicKWorldModel
from licwm.models.baselines.cfc_wm import CFCWorldModel
from licwm.models.baselines.moe_wm import MoEWorldModel

def build_model(cfg):
    d, m = cfg.domain, cfg.model
    name = m.name
    if name == "lic_wm":
        return LICWorldModel(d.obs_dim, d.action_dim, d.event_dim, m.h_fast, m.c_dim, m.num_prototypes, m.omega_max, use_event_token=m.use_event_token, transition_mode=m.transition_mode, stochastic_climate=m.stochastic_climate, enable_residual_channel=m.enable_residual_channel, residual_eps=m.residual_eps, fast_mode=m.fast_mode)
    if name == "gru_wm": return GRUWorldModel(d.obs_dim, d.action_dim, m.h_fast)
    if name == "context_wm": return ContextWorldModel(d.obs_dim, d.action_dim, m.context_dim, m.h_fast)
    if name == "transformer_wm": return TransformerWorldModel(d.obs_dim, d.action_dim, m.h_fast)
    if name == "auto_physick_wm": return AutoPhysicKWorldModel(d.obs_dim, d.action_dim)
    if name == "cfc_wm": return CFCWorldModel(d.obs_dim, d.action_dim, m.h_fast)
    if name == "moe_wm": return MoEWorldModel(d.obs_dim, d.action_dim, m.experts)
    raise ValueError(name)
