from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import hydra
from omegaconf import DictConfig

from ralagwm.training.paper_pipeline import (
    build_domain_cfg,
    load_training_presets,
    paper_training_jobs,
    summarize_final_metrics,
    train_audit_for_domain,
    train_baseline_for_domain,
    train_ralag_for_domain,
)
from ralagwm.evaluation.evaluator import load_manifest
from ralagwm.utils.io import dump_json, resolve_path
from ralagwm.utils.logging import setup_logging
from ralagwm.utils.progress import ProgressTracker

METHODS = ['ralag_wm', 'recon_wm', 'value_wm', 'policy_wm', 'rank_wm']


@hydra.main(version_base=None, config_path='../configs', config_name='base')
def main(cfg: DictConfig) -> None:
    setup_logging(str(getattr(cfg.logging, 'level', 'INFO')))
    manifest_path = getattr(getattr(cfg, 'experiment', {}), 'manifest_path', '') or 'inputs/manifests/paper_main_domains.yaml'
    manifest = load_manifest(manifest_path) or {}
    domains = manifest.get('domains', [])
    print(f'[paper-training] manifest={resolve_path(manifest_path)} domains={len(domains)}')
    if not domains:
        raise RuntimeError(f'No domains found in manifest: {resolve_path(manifest_path)}')
    presets = load_training_presets()
    tracker = ProgressTracker(total=paper_training_jobs(domains, METHODS))
    summary_rows = []
    for idx, domain in enumerate(domains):
        domain_cfg = build_domain_cfg(cfg, domain, presets)
        domain_label = domain.get('label', domain.get('name', f'domain{idx}'))
        audit_run = f'train-audit:{idx:02d}_{domain_label}'
        tracker.start(audit_run, group=domain.get('group', 'na'))
        try:
            audit_ckpt = train_audit_for_domain(domain_cfg, run_name=audit_run)
            tracker.done(audit_run, ckpt=audit_ckpt.name)
        except Exception as exc:
            tracker.fail(audit_run, exc)
            raise
        summary_rows.append({'run': audit_run, 'domain': domain_label, 'method': 'audit', 'checkpoint': str(audit_ckpt)})

        for method in METHODS:
            run_name = f'train-{method}:{idx:02d}_{domain_label}'
            tracker.start(run_name, group=domain.get('group', 'na'))
            try:
                if method == 'ralag_wm':
                    ckpt = train_ralag_for_domain(domain_cfg, run_name=run_name, audit_ckpt=audit_ckpt)
                else:
                    baseline_cfg = domain_cfg
                    baseline_cfg.baseline = method
                    ckpt = train_baseline_for_domain(baseline_cfg, run_name=run_name, baseline_name=method)
                tracker.done(run_name, ckpt=Path(ckpt).name)
            except Exception as exc:
                tracker.fail(run_name, exc)
                raise
            summary_rows.append({'run': run_name, 'domain': domain_label, 'method': method, 'checkpoint': str(ckpt)})
    dump_json('outputs/metrics/paper_training_summary.json', {'jobs': summary_rows})
    print('[paper-training] finished:', resolve_path('outputs/metrics/paper_training_summary.json'))


if __name__ == '__main__':
    main()
