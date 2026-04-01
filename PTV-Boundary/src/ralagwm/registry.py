"""Simple registry helpers."""
from typing import Callable, Dict
_REGISTRY: Dict[str, Callable] = {}
def register(name: str, fn: Callable) -> None:
    _REGISTRY[name] = fn
def get(name: str) -> Callable:
    return _REGISTRY[name]
