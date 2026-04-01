from __future__ import annotations
from copy import deepcopy
from licwm.training.engine import run_training
from licwm.evaluation.base import run_evaluation
from licwm.audits._common import write_audit_json


def _run(cfg, tag: str, use_climate_oracle: bool = False, use_law_oracle: bool = False):
    c = deepcopy(cfg)
    c.output_dir = f"outputs/runs/{tag}"
    outdir = run_training(c)
    c.checkpoint_path = f"{outdir}/checkpoint_best.pt"
    c.evaluator.name = "predictive"
    res = run_evaluation(c)
    res["oracle_mode"] = {"climate": use_climate_oracle, "law": use_law_oracle}
    return res


def run_oracle_audit(cfg):
    out = {
        "gt_climate_oracle": _run(cfg, "gt_climate_oracle", use_climate_oracle=True),
        "law_param_oracle": _run(cfg, "law_param_oracle", use_law_oracle=True),
    }
    return write_audit_json(cfg, 'oracle_runs', out)
