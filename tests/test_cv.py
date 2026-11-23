import numpy as np

from molecure.model.pls import Recipe
from molecure.validate import metrics
from molecure.validate.cv import (
    choose_components,
    cv_predictions,
    loo_folds,
    nested_loo,
    press_curve,
    q2_curve,
    random_folds,
    y_randomization,
)


def make_data(n=30, p=120, seed=0):
    """Like real field data: many correlated columns driven by a few hidden factors."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 3))
    X = Z @ rng.normal(size=(3, p)) + 0.2 * rng.normal(size=(n, p))
    y = Z @ np.array([1.0, -2.0, 0.5]) + 0.1 * rng.normal(size=n)
    return X, y


def plain(p):
    return Recipe(min_sigma=0.0, scaling="none", blocks=np.zeros(p, int))


def test_loo_holds_every_compound_out_exactly_once():
    folds = loo_folds(7)
    held_out = sorted(int(test[0]) for _, test in folds)
    assert held_out == list(range(7))
    assert all(len(train) == 6 and test[0] not in train for train, test in folds)


def test_random_folds_partition_the_compounds_evenly():
    folds = random_folds(21, 4, np.random.default_rng(0))
    tests = [sorted(test.tolist()) for _, test in folds]
    assert sorted(sum(tests, [])) == list(range(21))  # each compound held out once
    assert sorted(len(t) for t in tests) == [5, 5, 5, 6]  # like Cramer's "five or six per run"
    assert all(len(set(train) & set(test)) == 0 for train, test in folds)


def test_random_folds_are_reproducible_and_depend_on_the_seed():
    a = random_folds(21, 4, np.random.default_rng(5))
    b = random_folds(21, 4, np.random.default_rng(5))
    c = random_folds(21, 4, np.random.default_rng(6))
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b, strict=True))
    assert any(not np.array_equal(x[1], y[1]) for x, y in zip(a, c, strict=True))


def test_groups_stay_together():
    groups = np.repeat(np.arange(7), 3)  # 7 groups of 3 compounds
    for _, test in random_folds(21, 3, np.random.default_rng(1), groups):
        assert all(set(np.where(groups == g)[0]) <= set(test) for g in set(groups[test]))


def test_heldout_activity_cannot_influence_its_own_prediction():
    X, y = make_data()
    base = cv_predictions(X, y, plain(X.shape[1]), 3, loo_folds(len(y)))
    y_changed = y.copy()
    y_changed[0] = 1000.0
    changed = cv_predictions(X, y_changed, plain(X.shape[1]), 3, loo_folds(len(y)))
    assert np.array_equal(base[0], changed[0])  # compound 0 never saw its own activity


def test_the_sigma_filter_is_learned_inside_each_fold():
    X, y = make_data(p=40)
    X[:, 5] = 0.0
    X[0, 5] = 3.0  # varies ONLY because of compound 0
    recipe = Recipe(min_sigma=0.1, scaling="autoscale", blocks=np.zeros(40, int))
    first = cv_predictions(X, y, recipe, 2, loo_folds(len(y)))[0]
    X2 = X.copy()
    X2[0, 5] = 50.0
    second = cv_predictions(X2, y, recipe, 2, loo_folds(len(y)))[0]
    # when compound 0 is held out the column is constant in training, so it is dropped
    assert np.array_equal(first, second)


def test_a_real_signal_gives_a_high_q2_by_the_second_component():
    X, y = make_data()
    preds = cv_predictions(X, y, plain(X.shape[1]), 4, loo_folds(len(y)))
    assert q2_curve(y, preds)[1] > 0.7
    assert press_curve(y, preds).shape == (4,)


def test_selection_rules():
    n = 12
    e = 0.1 * (1 + np.arange(n) % 3)  # errors 0.1, 0.2, 0.3, ...
    y = np.zeros(n)
    preds = np.column_stack([np.full(n, 1.0), e, e * 0.99])  # the third component barely helps
    assert choose_components(y, preds, "min_press") == 3
    assert choose_components(y, preds, "one_se") == 2


def test_nested_loo_chooses_components_inside_each_fold():
    X, y = make_data(n=16, p=40)
    predictions, chosen = nested_loo(X, y, plain(40), 4, "min_press")
    assert predictions.shape == (16,) and chosen.shape == (16,)
    assert chosen.min() >= 1 and chosen.max() <= 4
    assert metrics.q2(y, predictions) > 0.5
    assert np.mean(chosen == 3) > 0.8  # the data have three hidden factors; selection finds them


def test_y_randomization_scores_far_below_the_real_model():
    X, y = make_data()
    real = metrics.q2(y, cv_predictions(X, y, plain(X.shape[1]), 2, loo_folds(len(y)))[:, 1])
    shuffled = y_randomization(X, y, plain(X.shape[1]), 2, 20, np.random.default_rng(0))
    assert shuffled.mean() < 0.1 and shuffled.max() < real
    again = y_randomization(X, y, plain(X.shape[1]), 2, 20, np.random.default_rng(0))
    assert np.array_equal(shuffled, again)
