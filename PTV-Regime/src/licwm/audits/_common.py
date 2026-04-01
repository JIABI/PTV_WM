from __future__ import annotations
import json
import os
from pathlib import Path

def _audit_meta(cfg, audit_name: str):
    return {
        'section': os.environ.get('LICWM_SECTION', 'manual'),
        'stage': 'audit',
        'domain': getattr(cfg.domain, 'name', 'unknown'),
        'task': getattr(cfg.task, 'name', audit_name),
        'audit': audit_name,
        'model': getattr(cfg.model, 'name', 'unknown'),
        'ablation': getattr(cfg.ablation, 'name', 'full'),
        'seed': int(getattr(cfg, 'seed', 0)),
        'run_index': os.environ.get('LICWM_RUN_INDEX', '0'),
    }


def write_audit_json(cfg, audit_name: str, payload):
    meta = _audit_meta(cfg, audit_name)
    obj = dict(meta)
    if isinstance(payload, dict):
        obj.update(payload)
    os.makedirs('outputs/aggregates', exist_ok=True)
    os.makedirs('outputs/aggregates/raw', exist_ok=True)
    legacy = Path('outputs/aggregates') / f'{audit_name}.json'
    legacy.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    raw = Path('outputs/aggregates/raw') / f"{meta['section']}__{audit_name}__{meta['domain']}__{meta['task']}__{meta['run_index']}.json"
    raw.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    return obj
