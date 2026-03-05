import numpy as np
import pandas as pd
import pytest

from hw5helper import (
    CVResult,
    GridSearchResult,
    SARIMAXParams,
    SARIMAXResult,
    Season,
    fit_prophet,
    fit_sarimax,
    forecast_prophet,
    forecast_sarimax,
    load_data,
    make_future_exog,
    rolling_cv,
    select_sarimax,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Small synthetic daily series with trend, weekly pattern, and exog."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    day_of_week = np.array([d.dayofweek for d in dates])
    weekly = 10 * np.sin(2 * np.pi * day_of_week / 7)
    trend = np.linspace(500, 550, n)
    marketing = np.random.uniform(80, 120, n)
    holiday = np.zeros(n, dtype=int)
    # mark a short holiday window
    holiday[150:160] = 1
    noise = np.random.normal(0, 5, n)
    sales = trend + weekly + 0.3 * marketing + 30 * holiday + noise

    df = pd.DataFrame(
        {"sales": sales, "marketing": marketing, "holiday": holiday},
        index=dates,
    )
    df.index.name = "date"
    df.index.freq = "D"
    return df


@pytest.fixture
def simple_params() -> SARIMAXParams:
    return SARIMAXParams(order=(1, 0, 0), seasonal_order=(0, 0, 0, 7))


# ---------------------------------------------------------------------------
# Season enum
# ---------------------------------------------------------------------------


class TestSeason:
    def test_values(self):
        assert Season.WEEKLY == 7
        assert Season.YEARLY == 365

    def test_usable_as_int(self):
        params = SARIMAXParams(order=(1, 0, 0), seasonal_order=(1, 0, 1, Season.WEEKLY))
        assert params.seasonal_order[3] == 7


# ---------------------------------------------------------------------------
# SARIMAXParams
# ---------------------------------------------------------------------------


class TestSARIMAXParams:
    def test_str(self):
        p = SARIMAXParams(order=(1, 1, 1), seasonal_order=(1, 0, 1, 7))
        assert str(p) == "SARIMAX(1, 1, 1)x(1, 0, 1, 7)"

    def test_equality(self):
        a = SARIMAXParams(order=(1, 0, 1), seasonal_order=(1, 1, 1, 7))
        b = SARIMAXParams(order=(1, 0, 1), seasonal_order=(1, 1, 1, 7))
        assert a == b


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------


class TestLoadData:
    def test_loads_csv(self, tmp_path):
        csv = tmp_path / "retail_sales.csv"
        dates = pd.date_range("2022-01-01", periods=10, freq="D")
        df = pd.DataFrame(
            {"date": dates, "sales": range(10), "marketing": range(10), "holiday": [0] * 10}
        )
        df.to_csv(csv, index=False)

        result = load_data(str(csv))
        assert result.index.name == "date"
        assert result.index.freq == "D"
        assert list(result.columns) == ["sales", "marketing", "holiday"]
        assert len(result) == 10


# ---------------------------------------------------------------------------
# make_future_exog
# ---------------------------------------------------------------------------


class TestMakeFutureExog:
    def test_shape_and_index(self, synthetic_df):
        exog = make_future_exog(synthetic_df, steps=30)
        assert len(exog) == 30
        assert list(exog.columns) == ["marketing", "holiday"]
        # first date should be day after last in df
        assert exog.index[0] == synthetic_df.index[-1] + pd.Timedelta(days=1)

    def test_binary_col_zeros(self, synthetic_df):
        exog = make_future_exog(synthetic_df, steps=10)
        assert (exog["holiday"] == 0).all()

    def test_numeric_col_mean(self, synthetic_df):
        exog = make_future_exog(synthetic_df, steps=10)
        expected_mean = synthetic_df["marketing"].mean()
        assert np.isclose(exog["marketing"].iloc[0], expected_mean)

    def test_custom_start(self, synthetic_df):
        exog = make_future_exog(synthetic_df, steps=5, start="2024-01-01")
        assert exog.index[0] == pd.Timestamp("2024-01-01")

    def test_custom_cols(self, synthetic_df):
        exog = make_future_exog(synthetic_df, steps=5, exog_cols=["marketing"])
        assert list(exog.columns) == ["marketing"]


# ---------------------------------------------------------------------------
# rolling_cv
# ---------------------------------------------------------------------------


class TestRollingCV:
    def test_returns_cv_result(self, synthetic_df, simple_params):
        endog = synthetic_df["sales"]
        exog = synthetic_df[["marketing", "holiday"]]
        result = rolling_cv(endog, exog, simple_params, k=2, horizon=10, verbose=False)

        assert isinstance(result, CVResult)
        assert len(result.fold_rmses) == 2
        assert result.avg_rmse == pytest.approx(np.mean(result.fold_rmses))
        assert result.params == simple_params

    def test_rmse_is_finite(self, synthetic_df, simple_params):
        endog = synthetic_df["sales"]
        exog = synthetic_df[["marketing", "holiday"]]
        result = rolling_cv(endog, exog, simple_params, k=2, horizon=10, verbose=False)
        assert np.isfinite(result.avg_rmse)
        assert all(np.isfinite(r) for r in result.fold_rmses)


# ---------------------------------------------------------------------------
# fit_sarimax / forecast_sarimax
# ---------------------------------------------------------------------------


class TestFitSARIMAX:
    def test_fit_returns_result(self, synthetic_df, simple_params):
        endog = synthetic_df["sales"]
        exog = synthetic_df[["marketing", "holiday"]]
        result = fit_sarimax(endog, exog, simple_params)

        assert isinstance(result, SARIMAXResult)
        assert result.params == simple_params
        assert result.cv_result is None

    def test_forecast_length(self, synthetic_df, simple_params):
        endog = synthetic_df["sales"]
        exog = synthetic_df[["marketing", "holiday"]]
        result = fit_sarimax(endog, exog, simple_params)

        future_exog = make_future_exog(synthetic_df, steps=10)
        forecast = forecast_sarimax(result, steps=10, future_exog=future_exog)
        assert len(forecast) == 10
        assert all(np.isfinite(forecast))


# ---------------------------------------------------------------------------
# select_sarimax
# ---------------------------------------------------------------------------


class TestSelectSARIMAX:
    def test_selects_best(self, synthetic_df):
        endog = synthetic_df["sales"]
        exog = synthetic_df[["marketing", "holiday"]]
        candidates = [
            SARIMAXParams(order=(1, 0, 0), seasonal_order=(0, 0, 0, 7)),
            SARIMAXParams(order=(1, 0, 1), seasonal_order=(0, 0, 0, 7)),
        ]
        result = select_sarimax(endog, exog, candidates, k=2, horizon=10, verbose=False)

        assert isinstance(result, GridSearchResult)
        assert result.best.params in candidates
        assert len(result.all_results) == 2
        assert result.best.cv_result is not None

    def test_raises_on_no_candidates(self, synthetic_df):
        endog = synthetic_df["sales"]
        exog = synthetic_df[["marketing", "holiday"]]
        with pytest.raises(ValueError, match="No valid SARIMAX"):
            select_sarimax(endog, exog, [], k=2, horizon=10, verbose=False)


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------


class TestProphet:
    def test_fit_prophet(self, synthetic_df):
        model = fit_prophet(
            synthetic_df,
            target_col="sales",
            regressor_cols=["marketing"],
            holiday_col="holiday",
        )
        assert "marketing" in model.extra_regressors

    def test_forecast_prophet_length(self, synthetic_df):
        model = fit_prophet(
            synthetic_df,
            target_col="sales",
            regressor_cols=["marketing"],
            holiday_col="holiday",
        )
        fc = forecast_prophet(model, synthetic_df, periods=15)
        assert len(fc) == 15
        assert "yhat" in fc.columns

    def test_forecast_prophet_with_defaults(self, synthetic_df):
        model = fit_prophet(
            synthetic_df,
            target_col="sales",
            regressor_cols=["marketing"],
            holiday_col="holiday",
        )
        fc = forecast_prophet(model, synthetic_df, periods=10, regressor_defaults={"marketing": 100.0})
        assert len(fc) == 10

    def test_fit_prophet_no_holidays(self, synthetic_df):
        model = fit_prophet(
            synthetic_df,
            target_col="sales",
            regressor_cols=["marketing"],
            holiday_col=None,
        )
        fc = forecast_prophet(model, synthetic_df, periods=5)
        assert len(fc) == 5
