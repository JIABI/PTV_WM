"""Optional import helpers.

These utilities make the real environment adapters usable even when optional
third-party packages are not installed. The adapters should fail with a clear
message only when construction is attempted, while the rest of the repository
remains importable.
"""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any


class OptionalDependencyError(ImportError):
    """Raised when an optional dependency is required at runtime."""


def optional_import(name: str) -> tuple[ModuleType | None, str | None]:
    """Attempt to import a module.

    Parameters
    ----------
    name:
        Module path, e.g. ``"gymnasium"`` or ``"dm_control.suite"``.

    Returns
    -------
    module, error:
        ``module`` is ``None`` on failure and ``error`` contains the import
        exception string.
    """
    try:
        module = import_module(name)
        return module, None
    except Exception as exc:  # pragma: no cover - exercised in env adapters
        return None, f"{type(exc).__name__}: {exc}"


def optional_import_attr(module_name: str, attr_name: str) -> tuple[Any | None, str | None]:
    """Attempt to import a specific attribute from a module."""
    module, err = optional_import(module_name)
    if module is None:
        return None, err
    try:
        return getattr(module, attr_name), None
    except Exception as exc:  # pragma: no cover - defensive
        return None, f"{type(exc).__name__}: {exc}"


def require_dependency(name: str, install_hint: str | None = None) -> ModuleType:
    """Import a dependency or raise a clear runtime error.

    Parameters
    ----------
    name:
        Module path to import.
    install_hint:
        Optional human-readable install note appended to the error.
    """
    module, err = optional_import(name)
    if module is None:
        hint = f" Install hint: {install_hint}" if install_hint else ""
        raise OptionalDependencyError(f"Required optional dependency '{name}' is unavailable: {err}.{hint}")
    return module


def dependency_available(name: str) -> bool:
    """Return whether a dependency can be imported."""
    module, _ = optional_import(name)
    return module is not None
