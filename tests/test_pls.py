import numpy as np
import pytest
from sklearn.cross_decomposition import PLSRegression

from molecure.model.pls import FittedPLS, ModelError, Recipe, fit_pls


def make_data(n=30, p=120, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = X[:, 0] - 2.0 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def recipe(p, scaling="none", min_sigma=0.0, blocks=None):
    blocks = np.zeros(p, int) if blocks is None else blocks
    return Recipe(min_sigma=min_sigma, scaling=scaling, blocks=blocks)


def test_matches_scikit_learn_for_every_number_of_components():
    X, y = make_data()
    ours = fit_pls(X, y, recipe(X.shape[1]), max_components=5).predict_all(X)
    for k in range(1, 6):
        theirs = PLSRegression(n_components=k, scale=False).fit(X, y).predict(X).ravel()
        assert np.allclose(ours[:, k - 1], theirs, atol=1e-9), k


def test_coefficients_in_original_units_reproduce_the_predictions():
    X, y = make_data()
    X[:, 60:90] *= 0.01  # flat columns, to be removed by the filter
    blocks = np.repeat([0, 1], X.shape[1] // 2)
    for scaling in ("none", "autoscale", "block"):
        model = fit_pls(X, y, recipe(X.shape[1], scaling, min_sigma=0.5, blocks=blocks), 4)
        for k in (1, 3):
            b, intercept = model.coefficients(k)
            assert np.allclose(X @ b + intercept, model.predict(X, k), atol=1e-9), (scaling, k)
            assert np.all(b[~model.keep] == 0)


def test_sigma_filter_removes_flat_and_constant_columns():
    X, y = make_data(p=20)
    X[:, :5] *= 0.01  # sigma about 0.01
    X[:, 5] = 3.0  # constant
    model = fit_pls(X, y, recipe(20, min_sigma=0.5), 3)
    assert not model.keep[:6].any() and model.keep[6:].all()
    # with no sigma filter only the constant column is lost
    assert fit_pls(X, y, recipe(20, min_sigma=0.0), 3).keep.sum() == 19


def test_autoscale_gives_every_kept_column_unit_variance():
    X, y = make_data(p=30)
    X *= np.linspace(0.5, 5, 30)
    model = fit_pls(X, y, recipe(30, "autoscale"), 3)
    assert np.allclose(model.scale * X[:, model.keep].std(axis=0, ddof=1), 1.0)


def test_block_scaling_gives_each_field_the_same_total_variance():
    X, y = make_data(p=40)
    X[:, 20:] *= 10  # the second block is ten times larger
    blocks = np.repeat([0, 1], 20)
    model = fit_pls(X, y, recipe(40, "block", blocks=blocks), 3)
    scaled_var = (model.scale * X[:, model.keep].std(axis=0, ddof=1)) ** 2
    kept_blocks = blocks[model.keep]
    assert scaled_var[kept_blocks == 0].sum() == pytest.approx(1.0)
    assert scaled_var[kept_blocks == 1].sum() == pytest.approx(1.0)


def test_stops_when_the_data_run_out_of_components():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(12, 2)) @ rng.normal(size=(2, 6))  # rank 2
    y = rng.normal(size=12)
    model = fit_pls(X, y, recipe(6), max_components=5)
    assert model.n_components <= 2
    assert model.predict_all(X).shape == (12, model.n_components)


def test_errors_are_clear():
    X, y = make_data(p=10)
    with pytest.raises(ModelError, match="no column"):
        fit_pls(X, y, recipe(10, min_sigma=100.0), 3)
    with pytest.raises(ModelError, match="scaling"):
        recipe(10, scaling="magic")
    with pytest.raises(ModelError, match="at least 3"):
        fit_pls(X[:2], y[:2], recipe(10), 1)
    with pytest.raises(ModelError, match="different number of columns"):
        fit_pls(X, y, recipe(7), 1)


def test_save_and_load_gives_identical_predictions():
    X, y = make_data()
    model = fit_pls(X, y, recipe(X.shape[1], "autoscale", min_sigma=0.2), 4)
    again = FittedPLS.from_arrays(model.to_arrays())
    assert np.array_equal(model.predict_all(X), again.predict_all(X))
