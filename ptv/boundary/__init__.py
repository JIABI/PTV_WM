"""PTV-Boundary integration namespace.

Wraps the migrated PTV-Boundary implementation currently housed in PTV-Boundary/src.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BOUNDARY_SRC = Path(__file__).resolve().parents[2] / "PTV-Boundary" / "src"
if _BOUNDARY_SRC.exists() and str(_BOUNDARY_SRC) not in sys.path:
    sys.path.insert(0, str(_BOUNDARY_SRC))

from ralagwm import *  # noqa: F401,F403
