"""Project-local site customization.

Ensures the `src/` layout is importable without an editable install during
smoke tests and local script execution.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Compatibility alias for older Gym code paths under NumPy >= 2.0
try:
    import numpy as np
    if not hasattr(np, "bool8"):
        np.bool8 = np.bool_
except Exception:
    pass
