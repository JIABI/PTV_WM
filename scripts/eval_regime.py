#!/usr/bin/env python
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "PTV-Regime"
SRC = PROJECT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.chdir(PROJECT)
runpy.run_path(str(PROJECT / "scripts" / "evaluate.py"), run_name="__main__")
