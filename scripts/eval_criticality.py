#!/usr/bin/env python
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "PTV-Criticality"
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(PROJECT)
runpy.run_path(str(SRC / "fatewm" / "experiments" / "toy_ablation" / "eval.py"), run_name="__main__")
