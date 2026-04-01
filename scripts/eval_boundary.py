#!/usr/bin/env python
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "PTV-Boundary"
SRC = PROJECT / "src"
for p in (PROJECT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.chdir(PROJECT)
runpy.run_path(str(PROJECT / "testing" / "eval_main_benchmark.py"), run_name="__main__")
