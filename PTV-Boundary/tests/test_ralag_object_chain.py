import torch

from ralagwm.audit.ensemble import AuditEnsemble
from ralagwm.audit.heads import build_audit_head_for_obs
from ralagwm.data.batch import Transition
from ralagwm.data.replay import ReplayBuffer
from ralagwm.models.ralag_wm import RALAGWM
from ralagwm.training.loops import build_ralag_batch_from_replay, train_ralag_step


def test_forward_builds_predicted_chart_and_selected_action():
    model = RALAGWM(obs_dim=16, action_dim=6, chart_mode='discrete', chart_budget=4, pool_budget=8)
    out = model(torch.randn(3, 16))
    assert out.pred_chart is not None
    assert out.pred_chart.actions.shape == (3, 4)
    assert out.pred_geometry is not None
    assert out.refined_geometry is not None
    assert out.selected_action.shape == (3,)


def test_replay_batch_contains_oracle_chart_state_and_geometry():
    model = RALAGWM(obs_dim=5, action_dim=1, chart_mode='continuous', chart_budget=4, pool_budget=8)
    audit = AuditEnsemble([build_audit_head_for_obs((5,), 8) for _ in range(2)])
    replay = ReplayBuffer(capacity=32)
    for _ in range(10):
        replay.add(Transition(obs=torch.randn(5), action=torch.randn(1), reward=0.0, next_obs=torch.randn(5), done=False))
    batch = build_ralag_batch_from_replay(replay, model, audit, batch_size=4)
    assert batch.chart_state is not None
    assert batch.chart is not None
    assert batch.geometry_target is not None
    assert batch.chart.actions.shape == (4, 4)
    assert batch.geometry_target.centered_scores.shape == (4, 4)



def test_continuous_chart_state_backward_no_inplace_error():
    model = RALAGWM(obs_dim=5, action_dim=1, chart_mode='continuous', chart_budget=4, pool_budget=8)
    audit = AuditEnsemble([build_audit_head_for_obs((5,), 8) for _ in range(2)])
    replay = ReplayBuffer(capacity=32)
    for _ in range(12):
        replay.add(Transition(obs=torch.randn(5), action=torch.randn(1), reward=0.0, next_obs=torch.randn(5), done=False))
    batch = build_ralag_batch_from_replay(replay, model, audit, batch_size=4)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = train_ralag_step(model, batch=batch, optimizer=optim)
    assert 'loss' in metrics
    assert metrics['loss'] >= 0.0
