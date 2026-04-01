from __future__ import annotations
import torch
from .base import PlannerBase


class CEMPlanner(PlannerBase):
    def __init__(self, cfg):
        self.horizon = cfg.horizon
        self.population = cfg.population
        self.elite = cfg.elite
        self.iters = cfg.iterations

    def _cost(self, traj: torch.Tensor) -> torch.Tensor:
        pos = traj[..., :2]
        goal_cost = -pos[..., 0].mean()
        pair = torch.cdist(pos[:, -1], pos[:, -1])
        collision = (pair < 0.15).float().mean()
        return goal_cost + 5.0 * collision

    def plan(self, model, batch, horizon: int | None = None):
        horizon = horizon or self.horizon
        b, _, n, a_dim = batch.action_hist.shape
        mean = torch.zeros(horizon, n, a_dim)
        std = torch.ones_like(mean) * 0.5
        best_seq = mean
        for _ in range(self.iters):
            samples = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(self.population, *mean.shape)
            costs = []
            for i in range(self.population):
                ah = batch.action_hist.clone()
                ah[:, :horizon] = samples[i].unsqueeze(0)
                out = model(batch.obs_hist, ah, batch.event_hist, horizon=horizon, teacher_forcing=False)
                costs.append(self._cost(out.pred_obs))
            costs = torch.stack(costs)
            elite_idx = costs.topk(self.elite, largest=False).indices
            elite = samples[elite_idx]
            mean, std = elite.mean(dim=0), elite.std(dim=0) + 1e-4
            best_seq = elite[0]
        ah = batch.action_hist.clone(); ah[:, :horizon] = best_seq.unsqueeze(0)
        return model(batch.obs_hist, ah, batch.event_hist, horizon=horizon, teacher_forcing=False).pred_obs
