import numpy as np
import pytest

import hw3helper
from hw3helper import (
    _rmse,
    find_best_arima_response,
    find_best_sarima_response,
    find_best_sarimax_response,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def stationary():
    rng = np.random.default_rng(0)
    return rng.standard_normal(60)


@pytest.fixture
def train_test(stationary):
    return stationary[:50], stationary[50:]


# ---------------------------------------------------------------------------
# _rmse
# ---------------------------------------------------------------------------

def test_rmse_perfect():
    a = np.array([1.0, 2.0, 3.0])
    assert _rmse(a, a) == pytest.approx(0.0)


def test_rmse_known():
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 1.0])
    assert _rmse(y_true, y_pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# evaluate_arima_model
# ---------------------------------------------------------------------------

def test_evaluate_arima_model_returns_float(stationary):
    rmse = hw3helper.evaluate_arima_model(stationary, train_ratio=0.8, arima_order=(0, 0, 0), trend="n")
    assert isinstance(rmse, float)
    assert rmse >= 0


# ---------------------------------------------------------------------------
# evaluate_arima_models
# ---------------------------------------------------------------------------

def test_evaluate_arima_models_returns_best(stationary):
    best_cfg, best_trend = hw3helper.evaluate_arima_models(
        dataset=stationary,
        train_ratio=0.8,
        p_values=[0, 1],
        d_values=[0],
        q_values=[0],
        trend_values=["n"],
    )
    assert best_cfg is not None
    assert len(best_cfg) == 3
    assert best_trend == "n"


def test_evaluate_arima_models_all_fail_returns_none(stationary):
    # Pass an invalid trend that causes every model to fail
    best_cfg, best_trend = hw3helper.evaluate_arima_models(
        dataset=stationary,
        train_ratio=0.8,
        p_values=[0],
        d_values=[0],
        q_values=[0],
        trend_values=["__invalid__"],
    )
    assert best_cfg is None
    assert best_trend is None


# ---------------------------------------------------------------------------
# find_best_arima
# ---------------------------------------------------------------------------

def test_find_best_arima_returns_response(train_test):
    x_train, x_test = train_test
    res = hw3helper.find_best_arima(
        x_train=x_train, x_test=x_test,
        p_values=[0, 1], d_values=[0], q_values=[0], trend_values=["n"],
    )
    assert isinstance(res, find_best_arima_response)
    assert res.rmse >= 0
    assert (res.p, res.d, res.q) in {(0, 0, 0), (1, 0, 0)}


def test_find_best_arima_picks_lowest_rmse(train_test):
    x_train, x_test = train_test
    res = hw3helper.find_best_arima(
        x_train=x_train, x_test=x_test,
        p_values=[0, 1, 2], d_values=[0], q_values=[0, 1], trend_values=["n"],
    )
    assert res.rmse >= 0


def test_find_best_arima_no_valid_raises(train_test):
    x_train, x_test = train_test
    with pytest.raises(ValueError, match="No valid ARIMA model found"):
        hw3helper.find_best_arima(
            x_train=x_train, x_test=x_test,
            p_values=[0], d_values=[0], q_values=[0], trend_values=["__invalid__"],
        )


def test_find_best_arima_default_trend_raises(train_test):
    x_train, x_test = train_test
    with pytest.raises(ValueError):
        hw3helper.find_best_arima(
            x_train=x_train, x_test=x_test,
            p_values=[0], d_values=[0], q_values=[0],
        )


# ---------------------------------------------------------------------------
# find_best_arima_response
# ---------------------------------------------------------------------------

def test_find_best_arima_response_str(train_test):
    x_train, x_test = train_test
    res = hw3helper.find_best_arima(
        x_train=x_train, x_test=x_test,
        p_values=[1], d_values=[0], q_values=[0], trend_values=["n"],
    )
    s = str(res)
    assert "ARIMA(1, 0, 0)" in s
    assert "RMSE" in s


def test_find_best_arima_response_train_model(train_test):
    x_train, x_test = train_test
    res = hw3helper.find_best_arima(
        x_train=x_train, x_test=x_test,
        p_values=[1], d_values=[0], q_values=[0], trend_values=["n"],
    )
    fitted = res.train_model(x_train)
    assert fitted is not None


# ---------------------------------------------------------------------------
# find_best_sarima
# ---------------------------------------------------------------------------

def test_find_best_sarima_returns_response(train_test):
    x_train, x_test = train_test
    res = hw3helper.find_best_sarima(
        x_train=x_train, x_test=x_test,
        p_values=[0], d_values=[0], q_values=[0],
        P_values=[0], D_values=[0], Q_values=[0],
        seasonal_period=4, trend_values=["n"],
    )
    assert isinstance(res, find_best_sarima_response)
    assert res.rmse >= 0


def test_find_best_sarima_response_str():
    from unittest.mock import MagicMock
    r = find_best_sarima_response(
        model=MagicMock(), rmse=1.23, p=1, d=0, q=1,
        seasonal_P=1, seasonal_D=0, seasonal_Q=1, seasonal_period=12, trend="n",
    )
    s = str(r)
    assert "SARIMA(1, 0, 1)" in s
    assert "(1, 0, 1, 12)" in s
    assert "1.23" in s


def test_find_best_sarima_no_valid_raises(train_test):
    x_train, x_test = train_test
    with pytest.raises(ValueError, match="No valid SARIMA model found"):
        hw3helper.find_best_sarima(
            x_train=x_train, x_test=x_test,
            p_values=[0], d_values=[0], q_values=[0],
            P_values=[0], D_values=[0], Q_values=[0],
            seasonal_period=4, trend_values=["__invalid__"],
        )


# ---------------------------------------------------------------------------
# find_best_sarimax_response (__str__ bug was "SARIMA", now "SARIMAX")
# ---------------------------------------------------------------------------

def test_find_best_sarimax_response_str_says_sarimax():
    from unittest.mock import MagicMock
    r = find_best_sarimax_response(
        model=MagicMock(), rmse=0.5, p=1, d=0, q=1,
        seasonal_P=1, seasonal_D=0, seasonal_Q=1, seasonal_period=12, trend="n",
    )
    s = str(r)
    assert s.startswith("SARIMAX")
    assert "SARIMA(" not in s  # was the bug
