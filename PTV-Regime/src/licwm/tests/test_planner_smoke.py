import torch
from omegaconf import OmegaConf
from licwm.models.builders import build_model
from licwm.planners.base import build_planner
from licwm.data.batch_types import MultiAgentBatch

def test_planner_smoke():
    cfg = OmegaConf.load('configs/config.yaml')
    cfg.model.name = 'lic_wm'
    cfg.trainer.history_len = 6
    cfg.trainer.pred_len = 3
    cfg.planner.name = 'mpc'
    m = build_model(cfg)
    b = MultiAgentBatch(obs_hist=torch.randn(1,6,4,4), action_hist=torch.randn(1,6,4,2), event_hist=torch.randn(1,6,5), fut_state=torch.randn(1,3,4,4), mask=torch.ones(1,6,4))
    out = build_planner(cfg).plan(m, b, horizon=2)
    assert out.shape[1] == 2
