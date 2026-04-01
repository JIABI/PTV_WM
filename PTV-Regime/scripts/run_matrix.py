import argparse
import os
import shlex
import subprocess
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.chdir(PROJECT_ROOT)
from omegaconf import OmegaConf

SCRIPT_BY_STAGE = {
    'train': 'scripts/train.py',
    'eval': 'scripts/evaluate.py',
    'audit': 'scripts/audit.py',
}


def _format_override(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    return str(v)


def _maybe_inject_checkpoint_bootstrap(stage: str, overrides: dict) -> dict:
    overrides = dict(overrides)
    if stage in {'eval', 'audit'} and 'checkpoint_path' not in overrides:
        overrides.setdefault('auto_train_if_missing_checkpoint', True)
    return overrides


def main():
    ap = argparse.ArgumentParser(description='Run a paper section/supplement sweep manifest.')
    ap.add_argument('--manifest', required=True, help='Path to a configs/sweep/*.yaml manifest')
    ap.add_argument('--python', default='python', help='Python executable')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    section = manifest_path.stem
    cfg = OmegaConf.load(manifest_path)
    runs = cfg.get('runs', [])
    if not runs:
        raise ValueError(f'No runs found in manifest {args.manifest}')
    for i, run in enumerate(runs):
        stage = run.get('stage', 'eval')
        script = SCRIPT_BY_STAGE[stage]
        overrides = _maybe_inject_checkpoint_bootstrap(stage, run.get('overrides', {}))
        cmd = [args.python, script]
        for k, v in overrides.items():
            cmd.append(f"{k}={_format_override(v)}")
        pretty = ' '.join(shlex.quote(c) for c in cmd)
        print(f"[{i+1}/{len(runs)}] {pretty}")
        if not args.dry_run:
            env = os.environ.copy()
            env['LICWM_SECTION'] = section
            env['LICWM_RUN_INDEX'] = str(i + 1)
            env['LICWM_STAGE'] = stage
            subprocess.run(cmd, check=True, env=env)

if __name__ == '__main__':
    main()
