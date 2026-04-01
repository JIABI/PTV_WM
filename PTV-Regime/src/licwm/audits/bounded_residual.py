from __future__ import annotations
from copy import deepcopy
from licwm.training.engine import run_training
from licwm.evaluation.base import run_evaluation
from licwm.audits._common import write_audit_json


def run_bounded_residual_audit(cfg):
    out = {}
    for name, enabled, eps in [("full", False, 0.0), ("bounded_residual", True, 0.05)]:
        c = deepcopy(cfg)
        c.model.enable_residual_channel = enabled
        c.model.residual_eps = eps
        c.output_dir = f"outputs/runs/{name}"
        outdir = run_training(c)
        c.checkpoint_path = f"{outdir}/checkpoint_best.pt"
        c.evaluator.name = "predictive"
        out[name] = run_evaluation(c)
    return write_audit_json(cfg, 'bounded_residual', out)
