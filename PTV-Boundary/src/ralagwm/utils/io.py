from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    candidate = repo_root() / p
    return candidate.resolve()


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_json(path: str | Path, obj: Any) -> None:
    path = resolve_path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: str | Path) -> Any:
    return json.loads(resolve_path(path).read_text())


def dump_yaml(path: str | Path, obj: Any) -> None:
    path = resolve_path(path)
    ensure_dir(path.parent)
    path.write_text(yaml.safe_dump(obj, sort_keys=False))


def load_yaml(path: str | Path) -> Any:
    return yaml.safe_load(resolve_path(path).read_text())
