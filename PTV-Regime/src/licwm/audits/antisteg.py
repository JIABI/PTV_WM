import json
from licwm.evaluation.base import run_evaluation
from licwm.audits._common import write_audit_json

def run_antisteg_audit(cfg):
    cfg.evaluator.name = "antisteg"
    result = run_evaluation(cfg)
    return write_audit_json(cfg, 'antisteg', result)
