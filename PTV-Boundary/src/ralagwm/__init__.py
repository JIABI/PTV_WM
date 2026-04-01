"""RALAG-WM package."""

import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from .typing import AuditScores, ChartState, BICChart, RALAGGeometry, RALAGBatch, ModelOutputs
