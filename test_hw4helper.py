import numpy as np
import pandas as pd
import pytest

import hw4helper
from hw4helper import CVResult, _to_series, _direct_forecast, _recursive_forecast


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    return pd.Series(rng.standard_normal(30), name="y")


@pytest.fixture
def dataframe(series):
    return series.to_frame()


# ---------------------------------------------------------------------------
# _to_series
# ---------------------------------------------------------------------------

def test_to_series_passthrough(series):
    result = _to_series(series)
    pd.testing.assert_series_equal(result, series)


def test_to_series_from_dataframe(dataframe):
    result = _to_series(dataframe)
    assert isinstance(result, pd.Series)
    assert result.name == dataframe.columns[0]
    assert len(result) == len(dataframe)


# ---------------------------------------------------------------------------
# _direct_forecast / _recursive_forecast
# ---------------------------------------------------------------------------

def test_direct_forecast_returns_per_step_mses(series):
    train, test = series[:20], series[20:23]
    mses = _direct_forecast(train, test, order=(1, 0, 0), forecast_horizon=3)
    assert len(mses) == 3
    assert all(m >= 0 for m in mses)


def test_recursive_forecast_returns_per_step_mses(series):
    train, test = series[:20], series[20:23]
    mses = _recursive_forecast(train, test, order=(1, 0, 0), forecast_horizon=3)
    assert len(mses) == 3
    assert all(m >= 0 for m in mses)


def test_recursive_forecast_does_not_mutate_train(series):
    train = series[:20].copy()
    original = train.copy()
    _recursive_forecast(train, series[20:23], order=(0, 0, 0), forecast_horizon=3)
    pd.testing.assert_series_equal(train, original)


# ---------------------------------------------------------------------------
# evaluate_arima_cv
# ---------------------------------------------------------------------------

def test_evaluate_arima_cv_returns_all_orders(series):
    results = hw4helper.evaluate_arima_cv(
        dataset=series,
        p_values=[0, 1],
        d_values=[0],
        q_values=[0, 1],
        forecast_horizon=2,
        n_folds=2,
    )
    orders = {r.order for r in results}
    assert orders == {(0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)}


def test_evaluate_arima_cv_result_shape(series):
    results = hw4helper.evaluate_arima_cv(
        dataset=series,
        p_values=[0],
        d_values=[0],
        q_values=[0],
        forecast_horizon=3,
        n_folds=2,
    )
    assert len(results) == 1
    r = results[0]
    assert r.order == (0, 0, 0)
    assert len(r.step_rmses) == 3
    assert all(v >= 0 for v in r.step_rmses)
    assert r.avg_rmse >= 0


def test_evaluate_arima_cv_accepts_dataframe(dataframe):
    results = hw4helper.evaluate_arima_cv(
        dataset=dataframe,
        p_values=[0],
        d_values=[0],
        q_values=[0],
        forecast_horizon=2,
        n_folds=2,
    )
    assert len(results) == 1


# ---------------------------------------------------------------------------
# evaluate_arima_recursive_cv
# ---------------------------------------------------------------------------

def test_evaluate_arima_recursive_cv_returns_all_orders(series):
    results = hw4helper.evaluate_arima_recursive_cv(
        dataset=series,
        p_values=[0, 1],
        d_values=[0],
        q_values=[0],
        forecast_horizon=2,
        n_folds=2,
    )
    orders = {r.order for r in results}
    assert orders == {(0, 0, 0), (1, 0, 0)}


def test_evaluate_arima_recursive_cv_result_shape(series):
    results = hw4helper.evaluate_arima_recursive_cv(
        dataset=series,
        p_values=[0],
        d_values=[0],
        q_values=[0],
        forecast_horizon=3,
        n_folds=2,
    )
    assert len(results) == 1
    r = results[0]
    assert len(r.step_rmses) == 3
    assert r.avg_rmse >= 0


def test_evaluate_arima_recursive_cv_accepts_dataframe(dataframe):
    results = hw4helper.evaluate_arima_recursive_cv(
        dataset=dataframe,
        p_values=[0],
        d_values=[0],
        q_values=[0],
        forecast_horizon=2,
        n_folds=2,
    )
    assert len(results) == 1


# ---------------------------------------------------------------------------
# print_best_models
# ---------------------------------------------------------------------------

def test_print_best_models_empty(capsys):
    hw4helper.print_best_models([])
    assert capsys.readouterr().out == ""


def test_print_best_models_selects_best(capsys):
    results = [
        CVResult(order=(1, 0, 0), step_rmses=[0.9, 0.5], avg_rmse=0.7),
        CVResult(order=(0, 0, 1), step_rmses=[0.4, 0.8], avg_rmse=0.6),
    ]
    hw4helper.print_best_models(results)
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert "(0, 0, 1)" in lines[0]   # best step 1
    assert "(1, 0, 0)" in lines[1]   # best step 2
    assert "(0, 0, 1)" in lines[2]   # best avg
