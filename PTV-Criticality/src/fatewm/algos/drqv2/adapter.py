"""Self-contained DrQ-v2-compatible fallback adapter."""

from fatewm.algos.minimal.adapter import MinimalAdapter


class DrQv2Adapter(MinimalAdapter):
    @property
    def is_model_based(self) -> bool:
        return True
