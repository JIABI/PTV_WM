"""Lightweight callbacks for logging hooks."""


class Callback:
    def on_step_end(self, step: int, metrics: dict[str, float]) -> None:
        return None
