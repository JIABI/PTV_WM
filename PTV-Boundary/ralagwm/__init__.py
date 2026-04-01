"""Compatibility shim for src-layout imports without installation."""
from __future__ import annotations

from pathlib import Path
import pkgutil

__path__ = pkgutil.extend_path(__path__, __name__)  # type: ignore[name-defined]
SRC_PKG = Path(__file__).resolve().parents[1] / "src" / "ralagwm"
if SRC_PKG.exists():
    __path__.append(str(SRC_PKG))  # type: ignore[attr-defined]
