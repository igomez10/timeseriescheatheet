import numpy as np
import pandas as pd
import pytest

from autoarimahelper import (
    AutoARIMAResult,
    EvalResult,
    evaluate_forecast,
    fit_auto_arima,
    forecast,
    plot_forecast,
    train_test_split,
    update_model,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def airline_series():
    """Airline passengers loaded from pmdarima, with DatetimeIndex."""
    import pmdarima as pm

    y = pm.datasets.load_airpassengers()
    dates = pd.date_range("1949-01", periods=len(y), freq="MS")
    return pd.Series(y, index=dates, name="Passengers")


@pytest.fixture
def small_series():
    """Small synthetic stationary series for fast tests."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    return pd.Series(rng.standard_normal(60), index=dates, name="y")


@pytest.fixture
def fitted_result(small_series):
    """A quickly-fitted auto_arima result on the small series."""
    return fit_auto_arima(
        small_series,
        seasonal=False,
        max_p=2,
        max_q=2,
        stepwise=True,
    )


# ---------------------------------------------------------------------------
# train_test_split
# ---------------------------------------------------------------------------


class TestTrainTestSplit:
    def test_lengths(self, airline_series):
        train, test = train_test_split(airline_series, n_test=24)
        assert len(train) + len(test) == len(airline_series)
        assert len(test) == 24

    def test_no_overlap(self, airline_series):
        train, test = train_test_split(airline_series, n_test=12)
        assert train.index[-1] < test.index[0]

    def test_values_preserved(self, airline_series):
        train, test = train_test_split(airline_series, n_test=12)
        recombined = pd.concat([train, test])
        pd.testing.assert_series_equal(recombined, airline_series)


# ---------------------------------------------------------------------------
# fit_auto_arima
# ---------------------------------------------------------------------------


class TestFitAutoArima:
    def test_returns_result(self, small_series):
        result = fit_auto_arima(
            small_series,
            seasonal=False,
            max_p=2,
            max_q=2,
        )
        assert isinstance(result, AutoARIMAResult)
        assert len(result.order) == 3
        assert len(result.seasonal_order) == 4
        assert np.isfinite(result.aic)

    def test_seasonal_model(self, airline_series):
        train = airline_series[:-24]
        result = fit_auto_arima(
            train,
            seasonal=True,
            m=12,
            max_p=2,
            max_q=2,
            max_P=1,
            max_Q=1,
            stepwise=True,
        )
        assert result.seasonal_order[3] == 12

    def test_bic_criterion(self, small_series):
        result = fit_auto_arima(
            small_series,
            seasonal=False,
            max_p=1,
            max_q=1,
            information_criterion="bic",
        )
        assert isinstance(result, AutoARIMAResult)

    def test_str(self, fitted_result):
        s = str(fitted_result)
        assert "ARIMA" in s
        assert "AIC=" in s

    def test_accepts_numpy_array(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(50)
        result = fit_auto_arima(
            y,
            seasonal=False,
            max_p=1,
            max_q=1,
        )
        assert isinstance(result, AutoARIMAResult)


# ---------------------------------------------------------------------------
# forecast
# ---------------------------------------------------------------------------


class TestForecast:
    def test_forecast_length(self, fitted_result):
        fc = forecast(fitted_result, n_periods=10)
        assert len(fc.values) == 10

    def test_forecast_conf_int_shape(self, fitted_result):
        fc = forecast(fitted_result, n_periods=5, return_conf_int=True)
        assert fc.conf_int is not None
        assert fc.conf_int.shape == (5, 2)

    def test_forecast_no_conf_int(self, fitted_result):
        fc = forecast(fitted_result, n_periods=5, return_conf_int=False)
        assert fc.conf_int is None
        assert len(fc.values) == 5

    def test_conf_int_lower_le_upper(self, fitted_result):
        fc = forecast(fitted_result, n_periods=10, return_conf_int=True)
        assert np.all(fc.conf_int[:, 0] <= fc.conf_int[:, 1])

    def test_forecast_values_finite(self, fitted_result):
        fc = forecast(fitted_result, n_periods=10)
        assert np.all(np.isfinite(fc.values))


# ---------------------------------------------------------------------------
# evaluate_forecast
# ---------------------------------------------------------------------------


class TestEvaluateForecast:
    def test_perfect_forecast(self):
        actual = np.array([1.0, 2.0, 3.0])
        result = evaluate_forecast(actual, actual)
        assert result.rmse == pytest.approx(0.0)
        assert result.mae == pytest.approx(0.0)

    def test_known_values(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([2.0, 3.0, 4.0])
        result = evaluate_forecast(actual, predicted)
        assert result.rmse == pytest.approx(1.0)
        assert result.mae == pytest.approx(1.0)

    def test_accepts_series(self):
        actual = pd.Series([10.0, 20.0, 30.0])
        predicted = np.array([11.0, 19.0, 31.0])
        result = evaluate_forecast(actual, predicted)
        assert isinstance(result, EvalResult)
        assert result.rmse > 0
        assert result.mae > 0

    def test_rmse_ge_mae(self):
        actual = np.array([0.0, 0.0, 0.0])
        predicted = np.array([1.0, 2.0, 3.0])
        result = evaluate_forecast(actual, predicted)
        assert result.rmse >= result.mae


# ---------------------------------------------------------------------------
# update_model
# ---------------------------------------------------------------------------


class TestUpdateModel:
    def test_update_changes_conf_int(self, fitted_result):
        fc_before = forecast(fitted_result, n_periods=3)
        new_obs = pd.Series([100.0, 200.0, 300.0])
        update_model(fitted_result, new_obs)
        fc_after = forecast(fitted_result, n_periods=3)
        # Even if point forecasts stay the same (e.g. ARIMA(0,0,0)),
        # the confidence interval width should change with extreme new obs.
        width_before = fc_before.conf_int[:, 1] - fc_before.conf_int[:, 0]
        width_after = fc_after.conf_int[:, 1] - fc_after.conf_int[:, 0]
        assert not np.allclose(width_before, width_after)

    def test_update_accepts_numpy(self, fitted_result):
        new_obs = np.array([1.0, 2.0])
        update_model(fitted_result, new_obs)
        fc = forecast(fitted_result, n_periods=3)
        assert len(fc.values) == 3


# ---------------------------------------------------------------------------
# plot_forecast
# ---------------------------------------------------------------------------


class TestPlotForecast:
    def test_returns_axes(self, small_series, fitted_result):
        train, test = train_test_split(small_series, n_test=10)
        fc = forecast(fitted_result, n_periods=10)
        ax = plot_forecast(train, test, fc, fc_index=test.index, title="Test")
        assert ax is not None
        plt_module = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_module.close("all")

    def test_no_test(self, small_series, fitted_result):
        fc = forecast(fitted_result, n_periods=5)
        ax = plot_forecast(small_series, None, fc, title="No test")
        assert ax is not None
        plt_module = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_module.close("all")

    def test_no_conf_int(self, small_series, fitted_result):
        fc = forecast(fitted_result, n_periods=5, return_conf_int=False)
        ax = plot_forecast(small_series, None, fc)
        assert ax is not None
        plt_module = __import__("matplotlib.pyplot", fromlist=["pyplot"])
        plt_module.close("all")

    def test_existing_axes(self, small_series, fitted_result):
        import matplotlib.pyplot as plt

        fig, existing_ax = plt.subplots()
        fc = forecast(fitted_result, n_periods=5)
        returned_ax = plot_forecast(small_series, None, fc, ax=existing_ax)
        assert returned_ax is existing_ax
        plt.close("all")
