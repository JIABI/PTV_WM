from __future__ import annotations
import numpy as np
import torch
from licwm.metrics.counterfactual import monotonicity_score


def _metric_from_prediction(pred_obs: torch.Tensor, metric_name: str) -> torch.Tensor:
    pos = pred_obs[..., :2]
    if metric_name == "min_separation":
        last = pos[:, -1]
        d = torch.cdist(last, last)
        eye = torch.eye(d.shape[-1], device=d.device)[None]
        d = d + 1e6 * eye
        return d.min(dim=-1).values.mean(dim=-1)
    if metric_name == "compactness":
        last = pos[:, -1]
        center = last.mean(dim=1, keepdim=True)
        return -((last - center) ** 2).sum(dim=-1).mean(dim=-1)
    if metric_name == "goal_progress":
        return pos[:, -1, :, 0].mean(dim=-1)
    return -torch.norm(pos, dim=-1).mean(dim=(1, 2))


def evaluate_counterfactual(model, loader, cfg):
    model.eval()
    deltas = np.linspace(cfg.evaluator.delta_min, cfg.evaluator.delta_max, cfg.evaluator.delta_steps)
    metric_name = getattr(cfg.evaluator, "metric_name", "min_separation")
    ys = []
    with torch.no_grad():
        batch = next(iter(loader))
        base_c = model.climate_encoder(batch.obs_hist, batch.action_hist, batch.event_hist)["c"]
        for d in deltas:
            override = base_c.clone()
            override[:, cfg.evaluator.climate_dim] += float(d)
            out = model(batch.obs_hist, batch.action_hist, batch.event_hist, horizon=cfg.trainer.pred_len, teacher_forcing=False, climate_oracle=override)
            ys.append(float(_metric_from_prediction(out.pred_obs, metric_name).mean()))
    return {"delta_sweep": deltas.tolist(), "response_curve": ys, "metric_name": metric_name, "monotonicity": monotonicity_score(deltas, ys)}
