"""PTV-Criticality integration namespace.

Wraps the migrated PTV-Criticality implementation currently housed in PTV-Criticality/src.
"""

from __future__ import annotations

import sys
from pathlib import Path

_CRITICALITY_SRC = Path(__file__).resolve().parents[2] / "PTV-Criticality" / "src"
if _CRITICALITY_SRC.exists() and str(_CRITICALITY_SRC) not in sys.path:
    sys.path.insert(0, str(_CRITICALITY_SRC))

from fatewm import *  # noqa: F401,F403
