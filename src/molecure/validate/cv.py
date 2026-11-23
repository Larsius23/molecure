"""Cross-validation machinery. Every fit goes through fit_pls on the TRAINING part of a fold only,
so the sigma filter, the scaling and the PLS components are learned without the held-out compounds.
"""

import math
from collections.abc import Sequence

import numpy as np

from molecure.model.pls import Recipe, fit_pls

from . import metrics

Fold = tuple[np.ndarray, np.ndarray]  # (indices to train on, indices held out)


def loo_folds(n: int) -> list[Fold]:
    """Leave-one-out: every compound is held out once."""
    return [(np.delete(np.arange(n), i), np.array([i])) for i in range(n)]


def random_folds(n: int, k: int, rng: np.random.Generator, groups=None) -> list[Fold]:
    """Random k-fold split. With `groups`, compounds of one group always stay together
    (use it for scaffold or cluster labels; without it every compound is its own group)."""
    groups = np.arange(n) if groups is None else np.asarray(groups)
    labels = np.unique(groups)
    if k < 2 or k > len(labels):
        raise ValueError(f"k = {k} needs between 2 and {len(labels)} groups")
    order = rng.permutation(labels)
    fold_of = {g: i % k for i, g in enumerate(order)}  # deal the groups out like cards
    fold = np.array([fold_of[g] for g in groups])
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]


def cv_predictions(
    X: np.ndarray, y: np.ndarray, recipe: Recipe, max_components: int, folds: Sequence[Fold]
) -> np.ndarray:
    """Cross-validated predictions for 1..max_components components: shape (n, max_components)."""
    out = np.full((len(y), max_components), np.nan)
    for train, test in folds:
        model = fit_pls(X[train], y[train], recipe, max_components)
        pred = model.predict_all(X[test])
        if pred.shape[1] < max_components:  # the data ran out of components: repeat the last
            padding = np.repeat(pred[:, -1:], max_components - pred.shape[1], axis=1)
            pred = np.hstack([pred, padding])
        out[test] = pred
    return out


def press_curve(y: np.ndarray, preds: np.ndarray) -> np.ndarray:
    return np.array([metrics.press(y, preds[:, k]) for k in range(preds.shape[1])])


def q2_curve(y: np.ndarray, preds: np.ndarray) -> np.ndarray:
    return np.array([metrics.q2(y, preds[:, k]) for k in range(preds.shape[1])])


def choose_components(y: np.ndarray, preds: np.ndarray, rule: str) -> int:
    """Number of components from a cross-validated prediction table.

    min_press : the number with the smallest PRESS (largest q2).
    one_se    : the smallest number whose PRESS is within one standard error of the minimum
                (a simpler model when the extra components buy nothing measurable).
    """
    curve = press_curve(y, preds)
    best = int(np.argmin(curve))
    if rule == "min_press":
        return best + 1
    if rule == "one_se":
        squared = (y - preds[:, best]) ** 2
        se = math.sqrt(len(y)) * float(np.std(squared, ddof=1))
        return int(np.where(curve <= curve[best] + se)[0][0]) + 1
    raise ValueError(f"selection rule must be 'min_press' or 'one_se', not {rule!r}")


def nested_loo(
    X: np.ndarray, y: np.ndarray, recipe: Recipe, max_components: int, rule: str
) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-out where the NUMBER OF COMPONENTS is also chosen inside each fold
    (by an inner leave-one-out on the training part). This is the honest estimate when
    the component count was selected from the data. Returns (predictions, chosen counts)."""
    n = len(y)
    predictions, chosen = np.empty(n), np.empty(n, dtype=int)
    for i in range(n):
        train = np.delete(np.arange(n), i)
        inner = cv_predictions(X[train], y[train], recipe, max_components, loo_folds(n - 1))
        k = choose_components(y[train], inner, rule)
        model = fit_pls(X[train], y[train], recipe, max_components)
        predictions[i] = model.predict(X[i : i + 1], k)[0]
        chosen[i] = k
    return predictions, chosen


def y_randomization(
    X: np.ndarray,
    y: np.ndarray,
    recipe: Recipe,
    n_components: int,
    runs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """q2 (leave-one-out) after shuffling the activities, `runs` times.
    A model that finds real structure must score far above these."""
    folds = loo_folds(len(y))
    scores = np.empty(runs)
    for r in range(runs):
        shuffled = rng.permutation(y)
        preds = cv_predictions(X, shuffled, recipe, n_components, folds)
        scores[r] = metrics.q2(shuffled, preds[:, n_components - 1])
    return scores
