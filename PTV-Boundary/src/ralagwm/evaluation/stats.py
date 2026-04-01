"""Statistical helpers for paper-facing reports.

These utilities implement light-weight summary statistics commonly used in
control and RL papers: mean, median, interquartile mean (IQM), and bootstrap
confidence intervals. They intentionally avoid heavyweight dependencies so the
repository remains easy to run in constrained environments.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


def _to_array(values: ArrayLike) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return arr[np.isfinite(arr)]


def mean_stat(values: ArrayLike) -> float:
    arr = _to_array(values)
    return float(arr.mean()) if arr.size else 0.0


def median_stat(values: ArrayLike) -> float:
    arr = _to_array(values)
    return float(np.median(arr)) if arr.size else 0.0


def iqm_stat(values: ArrayLike) -> float:
    """Interquartile mean.

    This matches the common robust aggregate used in RL benchmarks: sort the
    values, remove the lowest and highest quartiles, then average the middle
    half. For very small samples, falls back to the ordinary mean.
    """

    arr = np.sort(_to_array(values))
    n = arr.size
    if n == 0:
        return 0.0
    if n < 4:
        return float(arr.mean())
    lo = int(np.floor(0.25 * n))
    hi = int(np.ceil(0.75 * n))
    core = arr[lo:hi]
    return float(core.mean()) if core.size else float(arr.mean())


def bootstrap_ci(
    values: ArrayLike,
    stat_fn: Callable[[ArrayLike], float] = mean_stat,
    n_boot: int = 200,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    arr = _to_array(values)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        v = float(stat_fn(arr))
        return v, v
    rng = np.random.default_rng(seed)
    boots = []
    n = arr.size
    for _ in range(int(n_boot)):
        sample = arr[rng.integers(0, n, size=n)]
        boots.append(float(stat_fn(sample)))
    lo = float(np.quantile(boots, alpha / 2.0))
    hi = float(np.quantile(boots, 1.0 - alpha / 2.0))
    return lo, hi


def summarize_metric(values: ArrayLike, seed: int = 0) -> dict[str, float]:
    arr = _to_array(values)
    if arr.size == 0:
        return {
            "mean": 0.0,
            "median": 0.0,
            "iqm": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "num": 0.0,
        }
    lo, hi = bootstrap_ci(arr, stat_fn=mean_stat, seed=seed)
    return {
        "mean": mean_stat(arr),
        "median": median_stat(arr),
        "iqm": iqm_stat(arr),
        "ci95_low": lo,
        "ci95_high": hi,
        "num": float(arr.size),
    }


def human_normalized(score: float, random_score: float | None, human_score: float | None) -> float | None:
    if random_score is None or human_score is None:
        return None
    denom = float(human_score - random_score)
    if abs(denom) < 1e-8:
        return None
    return float((score - random_score) / denom)


def with_prefix(stats: dict[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in stats.items()}


def safe_rank_correlation(x: ArrayLike, y: ArrayLike) -> float:
    """Simple Spearman-like rank correlation without scipy.

    This is used for paper-facing summary tables where a deterministic, minimal
    dependency implementation is preferred over importing scipy.
    """

    ax = _to_array(x)
    ay = _to_array(y)
    n = min(ax.size, ay.size)
    if n < 2:
        return 0.0
    ax = ax[:n]
    ay = ay[:n]
    rx = np.argsort(np.argsort(ax))
    ry = np.argsort(np.argsort(ay))
    rx = rx.astype(np.float32)
    ry = ry.astype(np.float32)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = float(np.sqrt((rx ** 2).sum()) * np.sqrt((ry ** 2).sum()))
    if denom < 1e-8:
        return 0.0
    return float((rx * ry).sum() / denom)
