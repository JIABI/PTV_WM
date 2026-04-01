"""Self-contained DreamerV3-compatible fallback adapter.

This is intentionally lightweight: when an external DreamerV3 implementation is
not linked, the repository still exposes a fully runnable adapter with the same
unified interface required by the RRRM training code.
"""

from fatewm.algos.minimal.adapter import MinimalAdapter


class DreamerV3Adapter(MinimalAdapter):
    pass
