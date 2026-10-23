# -*- coding: utf-8 -*-
"""Split conformal：在 |log(true / estimate)| 残差上校准 + 覆盖率"""
import numpy as np

def score(true, base, corr):
    est = base * np.exp(corr)
    return np.abs(np.log(true) - np.log(np.clip(est, 1.0, None)))

def calibrate_q(scores_cal, alpha):
    scores_cal = np.asarray(scores_cal, dtype=float)
    if scores_cal.ndim != 1 or len(scores_cal) == 0 or not np.all(np.isfinite(scores_cal)):
        raise ValueError("calibration scores must be a nonempty finite vector")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    n = len(scores_cal)
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    # The (n+1)-st augmented score is +infinity. Capping at max(scores)
    # would overstate finite-sample coverage when calibration is too small.
    if rank > n:
        return float("inf")
    return float(np.sort(scores_cal)[rank - 1])

def coverage(scores, q):
    return float(np.mean(scores <= q))
