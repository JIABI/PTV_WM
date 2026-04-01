"""A tiny Hydra-compatible subset for local smoke tests.

Implements:
- initialize(...)
- compose(...)
- main(...)
with recursive YAML defaults loading and dotted CLI overrides.
"""
from __future__ import annotations

import contextlib
import functools
import inspect
import os
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

import yaml
from omegaconf import DictConfig, OmegaConf

_CONFIG_STACK: list[Path] = []


def _resolve_relative(base_file: Path, rel_path: str | None) -> Path:
    if rel_path is None:
        return base_file.parent
    rel = Path(rel_path)
    candidates = [
        (base_file.parent / rel).resolve(),
        (Path.cwd() / rel).resolve(),
        (Path.cwd() / rel.name).resolve(),
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0]


@contextlib.contextmanager
def initialize(version_base: str | None = None, config_path: str | None = None) -> Iterator[None]:
    del version_base
    caller_file = None
    for frame in inspect.stack()[1:]:
        filename = Path(frame.filename).resolve()
        if filename != Path(__file__).resolve():
            caller_file = filename
            break
    if caller_file is None:
        caller_file = Path.cwd() / "__main__.py"
    base = _resolve_relative(caller_file, config_path)
    _CONFIG_STACK.append(base)
    try:
        yield None
    finally:
        _CONFIG_STACK.pop()


def compose(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    if not _CONFIG_STACK:
        raise RuntimeError("hydra.initialize() must be active before compose().")
    root = _CONFIG_STACK[-1]
    cfg = _load_config(root, config_name)
    if overrides:
        cfg = _apply_overrides(cfg, overrides, root)
    return cfg


def main(version_base: str | None = None, config_path: str | None = None, config_name: str | None = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    del version_base

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        base_file = Path(inspect.getfile(fn)).resolve()
        cfg_root = _resolve_relative(base_file, config_path)
        name = config_name or "config"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            overrides = [arg for arg in sys.argv[1:] if "=" in arg]
            cfg = _load_config(cfg_root, name)
            if overrides:
                cfg = _apply_overrides(cfg, overrides, cfg_root)
            return fn(cfg, *args, **kwargs)

        return wrapper

    return decorator


def _load_yaml(path: Path) -> DictConfig:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return OmegaConf.create(data)


def _load_config(root: Path, name: str) -> DictConfig:
    target = (root / f"{name}.yaml").resolve()
    if not target.exists():
        raise FileNotFoundError(f"Config file not found: {target}")
    return _load_with_defaults(target, root)


def _load_with_defaults(file_path: Path, root: Path) -> DictConfig:
    raw = _load_yaml(file_path)
    defaults = raw.pop("defaults", []) if isinstance(raw, DictConfig) and "defaults" in raw else []
    merged = OmegaConf.create({})
    for entry in defaults:
        if entry == "_self_":
            continue
        if isinstance(entry, str):
            if entry == "_self_":
                continue
            child = _load_with_defaults(root / f"{entry}.yaml", root)
        elif isinstance(entry, DictConfig):
            key, value = next(iter(entry.items()))
            child = OmegaConf.create({key: _load_with_defaults(root / key / f"{value}.yaml", root)})
        elif isinstance(entry, dict):
            key, value = next(iter(entry.items()))
            child = OmegaConf.create({key: _load_with_defaults(root / key / f"{value}.yaml", root)})
        else:
            raise TypeError(f"Unsupported defaults entry: {entry!r}")
        merged = OmegaConf.merge(merged, child)
    merged = OmegaConf.merge(merged, raw)
    return merged


def _coerce_scalar(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        if "." in text:
            return float(text)
        return int(text)
    except ValueError:
        return text


def _apply_overrides(cfg: DictConfig, overrides: list[str], root: Path) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    for override in overrides:
        key, value_text = override.split("=", 1)
        # Hydra-style config group override, e.g. env=dummy
        if "." not in key and (root / key / f"{value_text}.yaml").exists():
            cfg[key] = _load_with_defaults(root / key / f"{value_text}.yaml", root)
            continue
        value = _coerce_scalar(value_text)
        cursor: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], DictConfig):
                cursor[part] = DictConfig()
            cursor = cursor[part]
        cursor[parts[-1]] = value
    return cfg
