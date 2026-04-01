from __future__ import annotations
from licwm.metrics.control import control_metrics
from licwm.metrics.recovery import recovery_latency
from licwm.metrics.tail_risk import cvar_tail
import torch


def evaluate_control(model, loader, planner, cfg):
    batch = next(iter(loader))
    plan = planner.plan(model, batch, horizon=cfg.planner.horizon)
    metrics = control_metrics(plan)
    risk = torch.cdist(plan[:, -1, :, :2], plan[:, -1, :, :2])
    metrics["cvar"] = cvar_tail((risk < 0.15).float())
    if batch.event_hist is not None:
        sep = torch.cdist(plan[:, :, :, :2].reshape(plan.shape[0], plan.shape[1], -1, 2)[:, :, 0], plan[:, :, :, :2].reshape(plan.shape[0], plan.shape[1], -1, 2)[:, :, 0])
        sep_metric = sep.mean(dim=(-1, -2)).squeeze(0)
        event_any = (batch.event_hist.sum(dim=-1) > 0).float().squeeze(0)
        metrics["recovery_latency"] = recovery_latency(sep_metric, event_any, threshold=float(sep_metric.mean()))
    return metrics
