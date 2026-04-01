"""Central typing exports for PTV-Boundary.

Prefer importing structured objects from here or from :mod:`ralagwm.data.batch`.
"""
from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import torch

from ralagwm.data.batch import AuditScores, BICChart, ChartState, ModelOutputs, RALAGBatch, RALAGGeometry, Transition

Tensor: TypeAlias = torch.Tensor
PathLike: TypeAlias = str | Path

__all__ = [
    'Tensor',
    'PathLike',
    'AuditScores',
    'ChartState',
    'BICChart',
    'RALAGGeometry',
    'RALAGBatch',
    'ModelOutputs',
    'Transition',
]
