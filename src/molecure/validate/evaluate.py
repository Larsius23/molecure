"""Fit one model and validate it: leave-one-out, repeated k-fold, y-randomization, external set.

Pure numpy: no RDKit, no config objects. The pipeline hands it a field table and activities.
"""

from dataclasses import dataclass

import numpy as np

from molecure.model.pls import FittedPLS, ModelError, Recipe, fit_pls

from . import metrics
from .cv import (
    choose_components,
    cv_predictions,
    loo_folds,
    nested_loo,
    q2_curve,
    random_folds,
    y_randomization,
)


@dataclass(frozen=True)
class EvalSettings:
    max_components: int = 8
    n_components: int | None = None  # fix it (e.g. to match a paper) or leave None to select
    selection: str = "min_press"
    loo: bool = True
    kfold_k: int = 5
    kfold_repeats: int = 20
    y_randomization_runs: int = 100
    nested: bool = True  # also report the honest leave-one-out with in-fold component choice
    seed: int = 42


@dataclass(frozen=True, eq=False)
class Evaluation:
    summary: dict  # JSON-able numbers
    model: FittedPLS  # fitted on ALL training compounds
    n_components: int  # the number actually used
    loo_predictions: np.ndarray | None  # (n_train,) cross-validated, at n_components
    fit_predictions: np.ndarray  # (n_train,) fitted values, at n_components
    external_predictions: np.ndarray | None  # (n_test,)


def _floats(values) -> list[float]:
    return [float(v) for v in values]


def evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    recipe: Recipe,
    settings: EvalSettings,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    groups: np.ndarray | None = None,
) -> Evaluation:
    X_train, y_train = np.asarray(X_train, float), np.asarray(y_train, float)
    n = len(y_train)
    A = min(settings.max_components, n - 2)
    if A < 1:
        raise ModelError(f"{n} training compounds are too few to fit a PLS model")
    rng = np.random.default_rng([settings.seed, 1])  # every random step derives from the seed
    summary: dict = {"n_train": n, "n_columns": int(X_train.shape[1])}

    # ---- cross-validation tables for 1..A components ----
    loo_preds = cv_predictions(X_train, y_train, recipe, A, loo_folds(n)) if settings.loo else None
    kfold_q2 = None
    if settings.kfold_repeats > 0:
        runs = []
        for _ in range(settings.kfold_repeats):
            folds = random_folds(n, settings.kfold_k, rng, groups)
            preds = cv_predictions(X_train, y_train, recipe, A, folds)
            runs.append(q2_curve(y_train, preds))
        kfold_q2 = np.array(runs)  # (repeats, A)

    # ---- number of components ----
    if settings.n_components is not None:
        k, rule = min(settings.n_components, A), "fixed"
    elif loo_preds is not None:
        k, rule = choose_components(y_train, loo_preds, settings.selection), settings.selection
    else:
        raise ModelError(
            "to choose the number of components, enable cv.loo or set pls.n_components"
        )
    summary["components"] = {"used": int(k), "rule": rule, "max": int(A)}

    # ---- the model on all training compounds ----
    model = fit_pls(X_train, y_train, recipe, A)
    k = min(k, model.n_components)
    fit_pred = model.predict(X_train, k)
    summary["n_columns_kept"] = int(model.keep.sum())
    summary["fit"] = {
        "r2": metrics.r2(y_train, fit_pred),
        "s": metrics.standard_error(y_train, fit_pred, k),
    }

    # ---- leave-one-out ----
    if loo_preds is not None:
        cv_at_k = loo_preds[:, k - 1]
        summary["loo"] = {
            "q2_by_components": _floats(q2_curve(y_train, loo_preds)),
            "q2": metrics.q2(y_train, cv_at_k),
            "spress": metrics.spress(y_train, cv_at_k, k),
        }
        if settings.nested and settings.n_components is None:
            nested_preds, chosen = nested_loo(X_train, y_train, recipe, A, settings.selection)
            summary["nested_loo"] = {
                "q2": metrics.q2(y_train, nested_preds),
                "chosen_components": [int(c) for c in chosen],
            }
    else:
        cv_at_k = None

    # ---- repeated k-fold ----
    if kfold_q2 is not None:
        at_k = kfold_q2[:, k - 1]
        summary["kfold"] = {
            "k": settings.kfold_k,
            "repeats": settings.kfold_repeats,
            "q2_mean": float(at_k.mean()),
            "q2_sd": float(at_k.std(ddof=1)) if len(at_k) > 1 else 0.0,
            "q2_min": float(at_k.min()),
            "q2_by_components_mean": _floats(kfold_q2.mean(axis=0)),
        }

    # ---- y-randomization ----
    if settings.y_randomization_runs > 0:
        scores = y_randomization(X_train, y_train, recipe, k, settings.y_randomization_runs, rng)
        summary["y_randomization"] = {
            "runs": settings.y_randomization_runs,
            "q2_mean": float(scores.mean()),
            "q2_max": float(scores.max()),
            "runs_at_or_above_real_q2": (
                int(np.sum(scores >= summary["loo"]["q2"])) if loo_preds is not None else None
            ),
        }

    # ---- external set ----
    ext_pred = None
    if X_test is not None and len(X_test) > 0:
        ext_pred = model.predict(np.asarray(X_test, float), k)
        if y_test is not None:
            summary["external"] = metrics.external_metrics(y_test, ext_pred, y_train)

    return Evaluation(
        summary=summary,
        model=model,
        n_components=int(k),
        loo_predictions=cv_at_k,
        fit_predictions=fit_pred,
        external_predictions=ext_pred,
    )
