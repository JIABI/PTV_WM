from __future__ import annotations
from copy import deepcopy
from licwm.training.engine import run_training
from licwm.evaluation.base import run_evaluation
from licwm.audits._common import write_audit_json


def run_ood_stress(cfg):
    out = {}
    settings = [("iid", 1.0), ("scaled_agents", 1.5)]
    for name, scale in settings:
        c = deepcopy(cfg)
        if hasattr(c.domain, "num_agents"):
            c.domain.num_agents = max(2, int(round(c.domain.num_agents * scale)))
        if hasattr(c.domain, "n_samples"):
            c.domain.n_samples = min(int(c.domain.n_samples), 1024)
        c.output_dir = f"outputs/runs/ood_{name}"
        outdir = run_training(c)
        c.checkpoint_path = f"{outdir}/checkpoint_best.pt"
        c.evaluator.name = "predictive"
        out[name] = run_evaluation(c)
    return write_audit_json(cfg, 'ood_stress', out)
