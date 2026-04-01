"""Planner registry for model-rollout control."""
from __future__ import annotations


class PlannerBase:
    def plan(self, model, batch, horizon: int):
        raise NotImplementedError


def build_planner(cfg):
    if cfg.planner.name == "cem":
        from .cem import CEMPlanner
        return CEMPlanner(cfg.planner)
    if cfg.planner.name == "mpc":
        from .mpc import MPCPlanner
        return MPCPlanner(cfg.planner)
    from .mpc import MPCPlanner
    return MPCPlanner(cfg.planner)
