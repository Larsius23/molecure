import json

import numpy as np
import pytest

from molecure.model.pls import ModelError, Recipe
from molecure.validate.evaluate import EvalSettings, evaluate


def make_split(n_train=24, n_test=6, p=100, seed=0):
    """Like real field data: many correlated columns driven by a few hidden factors."""
    rng = np.random.default_rng(seed)
    n = n_train + n_test
    Z = rng.normal(size=(n, 3))
    X = Z @ rng.normal(size=(3, p)) + 0.2 * rng.normal(size=(n, p))
    y = Z @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=n)
    return X[:n_train], y[:n_train], X[n_train:], y[n_train:]


def plain(p):
    return Recipe(min_sigma=0.0, scaling="none", blocks=np.zeros(p, int))


FAST = {"kfold_k": 4, "kfold_repeats": 5, "y_randomization_runs": 10, "max_components": 4}


def test_summary_has_everysection_for_a_fixed_number_of_components():
    Xtr, ytr, Xte, yte = make_split()
    result = evaluate(Xtr, ytr, plain(100), EvalSettings(n_components=2, **FAST), Xte, yte)
    s = result.summary
    assert s["components"] == {"used": 2, "rule": "fixed", "max": 4}
    assert s["loo"]["q2"] == pytest.approx(s["loo"]["q2_by_components"][1])
    assert 0 < s["fit"]["r2"] <= 1 and s["fit"]["s"] > 0
    assert s["kfold"]["repeats"] == 5 and "q2_mean" in s["kfold"]
    assert s["y_randomization"]["runs"] == 10
    assert s["external"]["n"] == 6 and len(result.external_predictions) == 6
    assert "nested_loo" not in s  # nothing was selected, so nothing to nest
    json.dumps(s)  # everything is plain, JSON-able numbers


def test_selecting_the_number_of_components_adds_the_honest_nested_estimate():
    Xtr, ytr, *_ = make_split()
    result = evaluate(Xtr, ytr, plain(100), EvalSettings(**FAST))
    s = result.summary
    assert 1 <= s["components"]["used"] <= 4 and s["components"]["rule"] == "min_press"
    assert len(s["nested_loo"]["chosen_components"]) == 24
    assert s["nested_loo"]["q2"] <= s["loo"]["q2_by_components"][s["components"]["used"] - 1] + 0.2


def test_same_seed_same_numbers_different_seed_different_kfold():
    Xtr, ytr, *_ = make_split()
    a = evaluate(Xtr, ytr, plain(100), EvalSettings(n_components=2, seed=1, **FAST)).summary
    b = evaluate(Xtr, ytr, plain(100), EvalSettings(n_components=2, seed=1, **FAST)).summary
    c = evaluate(Xtr, ytr, plain(100), EvalSettings(n_components=2, seed=2, **FAST)).summary
    assert json.dumps(a) == json.dumps(b)
    assert a["loo"] == c["loo"]  # leave-one-out has no randomness
    assert a["kfold"]["q2_mean"] != c["kfold"]["q2_mean"]


def test_no_external_section_without_an_external_set():
    Xtr, ytr, *_ = make_split()
    result = evaluate(Xtr, ytr, plain(100), EvalSettings(n_components=2, **FAST))
    assert "external" not in result.summary and result.external_predictions is None


def test_component_count_is_capped_by_the_number_of_compounds():
    Xtr, ytr, *_ = make_split(n_train=6)
    settings = EvalSettings(n_components=3, **{**FAST, "max_components": 8})
    result = evaluate(Xtr, ytr, plain(100), settings)
    assert result.summary["components"]["max"] == 4  # 6 compounds -> at most 4 components


def test_choosing_components_needs_leave_one_out():
    Xtr, ytr, *_ = make_split()
    with pytest.raises(ModelError, match="cv.loo"):
        evaluate(Xtr, ytr, plain(100), EvalSettings(loo=False, **FAST))
