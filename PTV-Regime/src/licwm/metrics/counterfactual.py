import numpy as np

def monotonicity_score(xs, ys):
    dy = np.diff(ys)
    return float((dy >= 0).mean() if np.mean(ys) >= 0 else (dy <= 0).mean())

def directional_consistency(base, altered):
    return float(np.sign(altered - base).mean())
