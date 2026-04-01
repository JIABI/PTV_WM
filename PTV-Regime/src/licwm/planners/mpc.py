from __future__ import annotations
import torch
from .base import PlannerBase


class MPCPlanner(PlannerBase):
    def __init__(self, cfg):
        self.horizon = getattr(cfg, "horizon", 8)
        self.iters = getattr(cfg, "iterations", 2)
        self.lr = getattr(cfg, "lr", 0.1)

    def _cost(self, traj: torch.Tensor) -> torch.Tensor:
        pos = traj[..., :2]
        goal_cost = -pos[..., 0].mean()
        pair = torch.cdist(pos[:, -1], pos[:, -1])
        collision = (pair < 0.15).float().mean()
        smooth = (traj[:, 1:] - traj[:, :-1]).pow(2).mean()
        return goal_cost + 5.0 * collision + 0.1 * smooth

    def plan(self, model, batch, horizon: int | None = None):
        horizon = horizon or self.horizon
        ah = batch.action_hist.clone()
        ah = ah[:, :max(horizon, ah.shape[1])].clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([ah], lr=self.lr)
        for _ in range(self.iters):
            out = model(batch.obs_hist, ah, batch.event_hist, horizon=horizon, teacher_forcing=False)
            cost = self._cost(out.pred_obs)
            opt.zero_grad(); cost.backward(); opt.step()
        return model(batch.obs_hist, ah.detach(), batch.event_hist, horizon=horizon, teacher_forcing=False).pred_obs.detach()
