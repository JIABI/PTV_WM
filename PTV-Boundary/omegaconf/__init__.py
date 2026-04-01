"""A tiny OmegaConf-compatible subset used for smoke testing.

This local shim exists because the execution environment used by tests may not
ship with hydra-core/omegaconf preinstalled. The API surface is intentionally
small but compatible with the project entrypoints.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


class DictConfig(dict):
    """Minimal recursive config with attribute access."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        incoming = dict(*args, **kwargs)
        for key, value in incoming.items():
            super().__setitem__(key, _wrap(value))

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, _wrap(value))

    def copy(self) -> "DictConfig":
        return DictConfig(super().copy())


def _wrap(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return value
    if isinstance(value, dict):
        return DictConfig(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _to_plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain(v) for v in value]
    return deepcopy(value)


class OmegaConf:
    """Tiny subset of the OmegaConf API used by this repo."""

    @staticmethod
    def create(data: Any) -> Any:
        return _wrap(data)

    @staticmethod
    def to_container(cfg: Any, resolve: bool = True) -> Any:
        del resolve
        return _to_plain(cfg)

    @staticmethod
    def merge(*configs: Any) -> DictConfig:
        out: DictConfig = DictConfig()
        for cfg in configs:
            out = _merge_two(out, _wrap(cfg))
        return out


@dataclass
class ListConfig(list):
    pass


def _merge_two(a: Any, b: Any) -> Any:
    if isinstance(a, DictConfig) and isinstance(b, DictConfig):
        merged = DictConfig(a)
        for key, value in b.items():
            if key in merged:
                merged[key] = _merge_two(merged[key], value)
            else:
                merged[key] = value
        return merged
    if isinstance(a, list) and isinstance(b, list):
        return [_wrap(v) for v in b]
    return _wrap(b)
