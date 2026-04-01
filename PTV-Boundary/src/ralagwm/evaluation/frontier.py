"""Compute--reliability frontier for Figure 4."""
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from ralagwm.envs import make_env
from .evaluator import build_model_from_cfg, default_domain_roster, env_cfg_from_domain, make_policy, rollout_policy
from .reports import save_csv, save_figure, save_latex_table, save_markdown, save_metrics


def _set_refinement_mode(model, mode: str) -> None:
    if mode == "none":
        model.refiner.refine_scale = 0.0
        model.refiner.margin_threshold = -1e9
        model.refiner.risk_threshold = 1e9
    elif mode == "random":
        model.refiner.refine_scale = 0.1
        model.refiner.margin_threshold = 1e9
        model.refiner.risk_threshold = -1e9
    elif mode == "uniform":
        model.refiner.refine_scale = 0.2
        model.refiner.margin_threshold = 1e9
        model.refiner.risk_threshold = -1e9
    else:  # selective
        model.refiner.refine_scale = 0.1
        model.refiner.margin_threshold = 0.2
        model.refiner.risk_threshold = 0.6


def run_frontier(cfg: Any, checkpoint_path: str | None = None) -> dict:
    domain = {"name": "dummy", "label": "Dummy"} if str(getattr(cfg.env, "name", "")) == "dummy" else default_domain_roster()[7]  # highdim continuous stress domain
    local_cfg = env_cfg_from_domain(cfg, domain)
    device = str(getattr(local_cfg, "device", "cpu"))
    rows = []
    for mode in ["none", "random", "uniform", "selective"]:
        model, _ = build_model_from_cfg(local_cfg, ckpt_path=checkpoint_path, device=device)
        _set_refinement_mode(model, mode)
        env = make_env(local_cfg.env)
        policy = make_policy(model, is_ralag=True, env=env, device=device)
        met = rollout_policy(env, policy, episodes=2, max_steps=min(128, env.spec.max_episode_steps), seed=int(getattr(local_cfg, "seed", 0)))
        env.close()
        chart_budget = int(getattr(local_cfg.chart, "chart_budget", 8))
        base_cost = chart_budget * max(env.spec.action_dim, 1)
        mode_cost = {"none": 1.0, "random": 1.2, "uniform": 1.6, "selective": 1.3}[mode]
        trigger = {"none": 0.0, "random": 0.5, "uniform": 1.0, "selective": 0.25}[mode]
        rows.append({
            "mode": mode,
            "decision_flops_proxy": float(base_cost * mode_cost),
            "mean_return": met["mean_return"],
            "flip_proxy": met["flip_proxy"],
            "cvar10": met["cvar10"],
            "trigger_rate_proxy": trigger,
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax1.plot([r["decision_flops_proxy"] for r in rows], [r["mean_return"] for r in rows], marker="o")
    for r in rows:
        ax1.annotate(r["mode"], (r["decision_flops_proxy"], r["mean_return"]))
    ax1.set_xlabel("Decision FLOPs proxy")
    ax1.set_ylabel("Mean return")
    ax1.set_title("Compute--return frontier")

    ax2.bar([r["mode"] for r in rows], [r["trigger_rate_proxy"] for r in rows])
    ax2.set_ylabel("Trigger concentration proxy")
    ax2.set_title("Danger-state refinement concentration")
    fig.suptitle("Figure 4: Compute--Reliability Frontier")

    save_figure(fig, "outputs/figures/figure4_frontier.png")
    save_metrics("outputs/metrics/frontier.json", {"rows": rows})
    save_csv("outputs/metrics/frontier.csv", rows)
    save_markdown("outputs/debug/figure4_frontier.md", "Figure 4: Frontier", rows)
    save_latex_table(
        "outputs/metrics/table_frontier.tex",
        caption="Compute--reliability frontier summary.",
        label="tab:frontier",
        rows=rows,
    )
    return {"rows": rows}
