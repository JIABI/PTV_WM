import numpy as np
from licwm.metrics.counterfactual import monotonicity_score

def test_monotonicity_score():
    x = np.array([-1, 0, 1])
    y = np.array([0.1, 0.2, 0.3])
    assert monotonicity_score(x, y) == 1.0
