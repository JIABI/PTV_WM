"""Shared PTV framework surface.

Phase-1 migration keeps atlas implementation in src/atlas_one_step and exposes
paper-level core entrypoints through this module.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from atlas_one_step.cli import (  # noqa: E402
    build_atlas_cli,
    evaluate_cli,
    fit_surrogate_cli,
    run_sweep,
    select_target_cli,
    smoke_test,
    train_cli,
)

__all__ = [
    "run_sweep",
    "build_atlas_cli",
    "fit_surrogate_cli",
    "select_target_cli",
    "train_cli",
    "evaluate_cli",
    "smoke_test",
]
