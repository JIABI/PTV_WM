"""Backward-compatible wrapper around the channelized, time-shared bank."""
from __future__ import annotations
from .channel_bank import ChannelPrototypeBank


class PrototypeBank(ChannelPrototypeBank):
    """Alias maintained for compatibility with older imports and checkpoints."""
    def __init__(self, n_channels: int, n_prototypes: int, mode: str = "mixed"):
        super().__init__(n_channels=n_channels, n_prototypes=n_prototypes, mode=mode)
