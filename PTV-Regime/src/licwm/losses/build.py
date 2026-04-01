from __future__ import annotations
import torch
from .step_loss import step_loss
from .rollout_loss import rollout_loss
from .slow_loss import slow_loss
from .jump_loss import jump_loss


def compute_loss(batch, out, cfg):
    target = batch.fut_state[:, : out.pred_obs.shape[1]]
    h = target.shape[1]
    h_weights = torch.linspace(1.0, 1.0 + 0.5 * max(h - 1, 0), steps=h, device=target.device)
    l_step = step_loss(out.pred_obs[:, :1], target[:, :1])
    l_roll = rollout_loss(out.pred_obs, target, horizon_weights=h_weights, event_hist=batch.event_hist, event_weight=cfg.trainer.event_weight)
    l_slow = slow_loss(out.aux["c_slow"], out.climate_states)
    l_jump = jump_loss(out.jump_gates)
    total = l_step + cfg.trainer.lambda_roll * l_roll + cfg.trainer.lambda_slow * l_slow + cfg.trainer.lambda_jump * l_jump
    return total, {"step": float(l_step.detach()), "roll": float(l_roll.detach()), "slow": float(l_slow.detach()), "jump": float(l_jump.detach()), "total": float(total.detach())}
