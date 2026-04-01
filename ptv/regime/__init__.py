"""PTV-Regime integration namespace.

Wraps the migrated LIC-WM implementation currently housed in PTV-Regime/src.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REGIME_SRC = Path(__file__).resolve().parents[2] / "PTV-Regime" / "src"
if _REGIME_SRC.exists() and str(_REGIME_SRC) not in sys.path:
    sys.path.insert(0, str(_REGIME_SRC))

from licwm import *  # noqa: F401,F403
