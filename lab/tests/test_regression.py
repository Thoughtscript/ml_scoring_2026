"""Regression scores: hand-computed from the residuals e = y - y_hat, asserted against scikit-learn."""
import numpy as np
import pytest
from sklearn.metrics import (explained_variance_score, max_error, mean_absolute_error,
                             mean_absolute_percentage_error, mean_squared_error,
                             median_absolute_error, r2_score, root_mean_squared_error,
                             root_mean_squared_log_error)

from common import scores
from common.scores import show

F = scores.REGRESSION_FORMULAS


def test_mae(regression):
    hand = float(regression["scores"]["mae"])
    ref = mean_absolute_error(regression["y"], regression["pred"])
    show("mae", F["mae"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_mse(regression):
    hand = float(regression["scores"]["mse"])
    ref = mean_squared_error(regression["y"], regression["pred"])
    show("mse", F["mse"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_rmse(regression):
    hand = float(regression["scores"]["rmse"])
    ref = root_mean_squared_error(regression["y"], regression["pred"])
    show("rmse", F["rmse"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_mape(regression):
    hand = float(regression["scores"]["mape"])
    ref = mean_absolute_percentage_error(regression["y"], regression["pred"])
    show("mape", F["mape"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_smape(regression):
    hand = float(regression["scores"]["smape"])
    y, y_hat = regression["y"], regression["pred"]
    ref = float(np.mean(2 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat))))
    show("smape", F["smape"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)
    assert 0 <= hand <= 2


def test_r2(regression):
    hand = float(regression["scores"]["r2"])
    ref = r2_score(regression["y"], regression["pred"])
    show("r2", F["r2"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_adjusted_r2(regression):
    hand = float(regression["scores"]["adjusted_r2"])
    n, p = len(regression["y"]), regression["p"]
    r2 = r2_score(regression["y"], regression["pred"])
    ref = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    show("adjusted_r2", F["adjusted_r2"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)
    assert hand < r2  # the penalty for p features always pulls it below R^2


def test_rmsle(regression):
    hand = float(regression["scores"]["rmsle"])
    ref = root_mean_squared_log_error(regression["y"], np.clip(regression["pred"], 0, None))
    show("rmsle", F["rmsle"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_explained_variance(regression):
    hand = float(regression["scores"]["explained_variance"])
    ref = explained_variance_score(regression["y"], regression["pred"])
    show("explained_variance", F["explained_variance"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_max_error(regression):
    hand = float(regression["scores"]["max_error"])
    ref = max_error(regression["y"], regression["pred"])
    show("max_error", F["max_error"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)


def test_median_ae(regression):
    hand = float(regression["scores"]["median_ae"])
    ref = median_absolute_error(regression["y"], regression["pred"])
    show("median_ae", F["median_ae"], hand, ref)
    assert hand == pytest.approx(ref, abs=1e-6)
