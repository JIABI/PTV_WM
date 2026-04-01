from typing import Any, Callable

class Registry:
    def __init__(self) -> None:
        self._items: dict[str, Callable[..., Any]] = {}

    def register(self, name: str, fn: Callable[..., Any]) -> None:
        if name in self._items:
            raise KeyError(name)
        self._items[name] = fn

    def build(self, name: str, *args, **kwargs):
        return self._items[name](*args, **kwargs)
