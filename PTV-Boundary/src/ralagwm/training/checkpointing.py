from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    extra: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """
    Save a training checkpoint.

    Contents:
        - model_state
        - optimizer_state (optional)
        - extra
        - config
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "extra": extra or {},
        "config": config or {},
    }

    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()

    torch.save(payload, path)
    return path


def _is_uninitialized_tensor(obj: Any) -> bool:
    return isinstance(obj, (UninitializedParameter, UninitializedBuffer))


def _safe_shape(obj: Any) -> tuple[int, ...] | None:
    """
    Return tensor shape if safely accessible, else None.
    """
    if _is_uninitialized_tensor(obj):
        return None
    try:
        return tuple(obj.shape)
    except Exception:
        return None


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint into model / optimizer.

    Behavior:
        - strict=True: standard load_state_dict
        - strict=False:
            * only load keys present in current model
            * if current parameter is initialized, require shape match
            * if current parameter is uninitialized (Lazy modules), allow loading
    """
    ckpt = torch.load(Path(path), map_location=map_location)
    state = ckpt.get("model_state", ckpt)

    if strict:
        model.load_state_dict(state, strict=True)
    else:
        current = model.state_dict()
        filtered: dict[str, Any] = {}

        for k, v in state.items():
            if k not in current:
                continue

            cur_tensor = current[k]
            cur_shape = _safe_shape(cur_tensor)
            new_shape = _safe_shape(v)

            # If current tensor is uninitialized (Lazy module), allow load directly.
            if cur_shape is None:
                filtered[k] = v
                continue

            # If checkpoint tensor shape isn't safely readable, skip conservatively.
            if new_shape is None:
                continue

            if cur_shape == new_shape:
                filtered[k] = v

        model.load_state_dict(filtered, strict=False)

    if optimizer is not None and "optimizer_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        except Exception:
            # Keep training usable even if optimizer state is incompatible.
            pass

    return {
        "extra": ckpt.get("extra", {}),
        "config": ckpt.get("config", {}),
    }