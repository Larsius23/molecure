"""Each statistic checked against a hand-worked value; definitions checked against the papers."""

import math

import numpy as np
import pytest

from molecure.validate import metrics

# Coats 1998, Table 1: affinity constants K x 1e6 (None = "< 0.1", listed as log K = 5)
CBG_K = [
    1.9, None, None, 0.58, 0.41, 76, 76, 7.8, None, 45, 76,
    0.83, None, None, None, 0.18, 0.18, None, 24, 55, 5.3,
]  # fmt: skip
TEBG_K = [
    0.21, 1300, 1500, 29, 14, 2.2, 1.6, 2.7, 66, 24, 16,
    5500, 680, 4.3, 150, 1.4, 14, 2.3, 8.8, 9.9, 1600,
]  # fmt: skip


def log_k(values):
    return np.array([5.0 if k is None else math.log10(k * 1e6) for k in values])


def test_q2_by_hand():
    y, y_cv = [1, 2, 3, 4], [1.5, 2, 3, 3.5]  # PRESS = 0.25 + 0 + 0 + 0.25 ; SS_tot = 5
    assert metrics.press(y, y_cv) == pytest.approx(0.5)
    assert metrics.q2(y, y_cv) == pytest.approx(0.9)


def test_q2_is_negative_when_the_model_is_worse_than_the_mean():
    assert metrics.q2([1, 2, 3], [3, 2, 1]) < 0


def test_r2_uses_the_same_formula_on_fitted_values():
    assert metrics.r2([1, 2, 3, 4], [1.5, 2, 3, 3.5]) == pytest.approx(0.9)


def test_standard_errors_divide_by_n_minus_c_minus_1():
    y, fit = [1, 2, 3, 4], [1.5, 2, 3, 3.5]  # sum of squares 0.5, n = 4, c = 1 -> dof 2
    assert metrics.spress(y, fit, 1) == pytest.approx(0.5)
    assert metrics.standard_error(y, fit, 1) == pytest.approx(0.5)


def test_standard_error_is_nan_without_degrees_of_freedom():
    assert math.isnan(metrics.spress([1, 2, 3], [1, 2, 3.5], 2))


def test_definitions_reproduce_cramers_table_ii():
    # Table II: q2 and the standard error of the cross-validated predictions ("press").
    # Build predictions whose PRESS is exactly (1 - q2) * SS_tot and compare.
    for k_values, q2_paper, press_paper in ((CBG_K, 0.662, 0.719), (TEBG_K, 0.555, 0.849)):
        y = log_k(k_values)
        error = math.sqrt((1 - q2_paper) * metrics.total_ss(y) / len(y))
        y_cv = y + error * np.where(np.arange(len(y)) % 2 == 0, 1.0, -1.0)
        assert metrics.q2(y, y_cv) == pytest.approx(q2_paper, abs=1e-9)
        assert metrics.spress(y, y_cv, n_components=2) == pytest.approx(press_paper, abs=0.002)


def test_q2_f1_and_f2_differ_in_their_reference_mean():
    y_ext, pred, y_train = [2, 4], [3, 3], [-1, 1]  # training mean 0, external mean 3
    assert metrics.q2_f1(y_ext, pred, y_train) == pytest.approx(0.9)  # 1 - 2 / (4 + 16)
    assert metrics.q2_f2(y_ext, pred) == pytest.approx(0.0)  # 1 - 2 / 2


def test_q2_f3_compares_per_compound_errors_with_training_variance():
    # per-compound external error 2/2 = 1 ; per-compound training variance 2/2 = 1
    assert metrics.q2_f3([2, 4], [3, 3], [-1, 1]) == pytest.approx(0.0)


def test_ccc_is_one_for_identical_and_penalises_a_shift():
    assert metrics.ccc([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert metrics.ccc([1, 2, 3], [2, 3, 4]) == pytest.approx(4 / 7)  # 2*(2/3) / (2/3 + 2/3 + 1)


def test_r2_corr_ignores_scale_and_offset():
    assert metrics.r2_corr([1, 2, 3, 4], [3, 5, 7, 9]) == pytest.approx(1.0)
    assert math.isnan(metrics.r2_corr([1, 2], [1, 2]))  # too few points to say


def test_external_metrics_bundle():
    result = metrics.external_metrics([2, 4, 6], [2, 4, 6], [1, 2, 3, 4, 5, 6])
    assert result["n"] == 3 and result["rmse"] == pytest.approx(0.0)
    assert result["q2_f1"] == pytest.approx(1.0) and result["ccc"] == pytest.approx(1.0)
