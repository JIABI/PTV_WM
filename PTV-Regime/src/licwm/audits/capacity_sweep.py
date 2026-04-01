from __future__ import annotations
from copy import deepcopy
from licwm.training.engine import run_training
from licwm.evaluation.base import run_evaluation
from licwm.audits._common import write_audit_json


def run_capacity_sweep(cfg):
    out = {}
    for size, n in [("small", 2), ("medium", 4), ("large", 8)]:
        c = deepcopy(cfg)
        c.model.num_prototypes = n
        if hasattr(c.domain, "n_samples"):
            c.domain.n_samples = min(int(c.domain.n_samples), 1024)
        c.output_dir = f"outputs/runs/capacity_{size}"
        outdir = run_training(c)
        c.checkpoint_path = f"{outdir}/checkpoint_best.pt"
        c.evaluator.name = "predictive"
        out[size] = run_evaluation(c)
    return write_audit_json(cfg, 'capacity_sweep', out)
