"""Paper-aligned evaluation modules."""

from .benchmark import run_benchmark
from .matched_fidelity import run_matched_fidelity
from .oracle_substitution import run_oracle_substitution
from .robustness import run_robustness
from .frontier import run_frontier

__all__ = [
    "run_benchmark",
    "run_matched_fidelity",
    "run_oracle_substitution",
    "run_robustness",
    "run_frontier",
]
