from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    cmd = [sys.executable, script]
    print('[paper-all] running', ' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    run('training/run_paper_training.py')
    run('testing/run_paper_main.py')
    run('testing/run_paper_si.py')
