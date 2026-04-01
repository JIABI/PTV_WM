from __future__ import annotations
from copy import deepcopy
from licwm.training.engine import run_training
from licwm.evaluation.base import run_evaluation
from licwm.audits._common import write_audit_json


def run_no_event_audit(cfg):
    c1 = deepcopy(cfg)
    c1.model.use_event_token = True
    d1 = run_training(c1)
    c1.checkpoint_path = f"{d1}/checkpoint_best.pt"
    c1.evaluator.name = "predictive"
    with_event = run_evaluation(c1)

    c2 = deepcopy(cfg)
    c2.model.use_event_token = False
    d2 = run_training(c2)
    c2.checkpoint_path = f"{d2}/checkpoint_best.pt"
    c2.evaluator.name = "predictive"
    no_event = run_evaluation(c2)
    return write_audit_json(cfg, 'no_event', {"with_event": with_event, "no_event": no_event})
