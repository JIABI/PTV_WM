from __future__ import annotations
from copy import deepcopy
from licwm.training.engine import run_training
from licwm.evaluation.base import run_evaluation
from licwm.audits._common import write_audit_json


def run_leakage_audit(cfg):
    out = {}
    for name, eps in [("full", 0.0), ("leakage", 0.2)]:
        c = deepcopy(cfg)
        c.model.enable_residual_channel = name == "leakage"
        c.model.residual_eps = eps
        if hasattr(c.domain, "n_samples"):
            c.domain.n_samples = min(int(c.domain.n_samples), 1024)
        c.output_dir = f"outputs/runs/{name}_audit"
        outdir = run_training(c)
        c.checkpoint_path = f"{outdir}/checkpoint_best.pt"
        c.evaluator.name = "antisteg"
        out[name] = run_evaluation(c)
    return write_audit_json(cfg, 'leakage', out)
