"""Validation statistics, each a small function you can check by hand.

Conventions (checked against the CoMFA papers, whose numbers they reproduce):
  q2        = 1 - PRESS / SS_tot            PRESS = sum of squared CROSS-VALIDATED errors
  r2        = 1 - SSE / SS_tot              SSE   = sum of squared errors of the FITTED values
  spress    = sqrt(PRESS / (n - c - 1))     "standard error of prediction"  (c = components)
  s         = sqrt(SSE   / (n - c - 1))     "standard error of the estimate"
SS_tot is taken about the mean of the training activities.

External-set statistics: the literature has several definitions of "R2_pred".
We report them all, by their usual names (Consonni et al. 2009):
  q2_f1  reference = mean of the TRAINING set   (the one Cramer 1988 used)
  q2_f2  reference = mean of the external set
  q2_f3  errors per compound vs training variance per compound
  ccc    Lin's concordance correlation coefficient
  r2_corr   squared Pearson correlation (observed vs predicted)
"""

import math

import numpy as np


def _arrays(*values):
    return [np.asarray(v, dtype=float) for v in values]


def press(y, y_cv) -> float:
    y, y_cv = _arrays(y, y_cv)
    return float(np.sum((y - y_cv) ** 2))


def total_ss(y, mean: float | None = None) -> float:
    (y,) = _arrays(y)
    centre = y.mean() if mean is None else mean
    return float(np.sum((y - centre) ** 2))


def q2(y, y_cv) -> float:
    """Cross-validated r2: 1 - PRESS / SS_tot."""
    return 1.0 - press(y, y_cv) / total_ss(y)


def r2(y, y_fit) -> float:
    """Conventional (fitted) r2."""
    return 1.0 - press(y, y_fit) / total_ss(y)


def _dof_root(sum_squares: float, n: int, n_components: int) -> float:
    dof = n - n_components - 1
    return math.sqrt(sum_squares / dof) if dof > 0 else math.nan


def spress(y, y_cv, n_components: int) -> float:
    return _dof_root(press(y, y_cv), len(y), n_components)


def standard_error(y, y_fit, n_components: int) -> float:
    return _dof_root(press(y, y_fit), len(y), n_components)


# ---------- external set ----------


def q2_f1(y_ext, y_pred, y_train) -> float:
    return 1.0 - press(y_ext, y_pred) / total_ss(y_ext, mean=float(np.mean(y_train)))


def q2_f2(y_ext, y_pred) -> float:
    return 1.0 - press(y_ext, y_pred) / total_ss(y_ext)


def q2_f3(y_ext, y_pred, y_train) -> float:
    per_ext = press(y_ext, y_pred) / len(y_ext)
    per_train = total_ss(y_train) / len(y_train)
    return 1.0 - per_ext / per_train


def ccc(y, y_pred) -> float:
    y, y_pred = _arrays(y, y_pred)
    covariance = float(np.mean((y - y.mean()) * (y_pred - y_pred.mean())))
    denominator = y.var() + y_pred.var() + (y.mean() - y_pred.mean()) ** 2
    return 2.0 * covariance / float(denominator) if denominator > 0 else math.nan


def r2_corr(y, y_pred) -> float:
    y, y_pred = _arrays(y, y_pred)
    if len(y) < 3 or y.std() == 0 or y_pred.std() == 0:
        return math.nan
    return float(np.corrcoef(y, y_pred)[0, 1] ** 2)


def external_metrics(y_ext, y_pred, y_train) -> dict[str, float]:
    """Every external-set statistic for one set of predictions."""
    y_ext, y_pred, y_train = _arrays(y_ext, y_pred, y_train)
    return {
        "n": int(len(y_ext)),
        "rmse": math.sqrt(press(y_ext, y_pred) / len(y_ext)),
        "q2_f1": q2_f1(y_ext, y_pred, y_train),
        "q2_f2": q2_f2(y_ext, y_pred),
        "q2_f3": q2_f3(y_ext, y_pred, y_train),
        "ccc": ccc(y_ext, y_pred),
        "r2_corr": r2_corr(y_ext, y_pred),
    }
