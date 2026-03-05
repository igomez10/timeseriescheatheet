from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import root_mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper


class Season(IntEnum):
    """Common seasonal periods for time series."""

    DAILY = 1
    WEEKLY = 7
    MONTHLY = 30
    YEARLY = 365


@dataclass
class SARIMAXParams:
    """SARIMAX (p,d,q)x(P,D,Q,s) parameter set."""

    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]

    def __str__(self) -> str:
        return f"SARIMAX{self.order}x{self.seasonal_order}"


@dataclass
class CVResult:
    """Result from rolling cross-validation."""

    avg_rmse: float
    fold_rmses: list[float]
    params: SARIMAXParams


@dataclass
class SARIMAXResult:
    """Result from fitting a SARIMAX model."""

    fitted: object  # SARIMAXResultsWrapper
    params: SARIMAXParams
    cv_result: CVResult | None = None


@dataclass
class GridSearchResult:
    """Result from SARIMAX grid search."""

    best: SARIMAXResult
    all_results: list[CVResult] = field(default_factory=list)
    failed: int = 0


@dataclass
class ProphetResult:
    """Result from fitting and forecasting with Prophet."""

    model: Prophet
    forecast: pd.DataFrame


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(path: str = "retail_sales.csv") -> pd.DataFrame:
    """Load retail sales CSV with date index and daily frequency."""
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date")
    df.index.freq = "D"
    return df


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def rolling_cv(
    endog: pd.Series,
    exog: pd.DataFrame,
    params: SARIMAXParams,
    k: int = 5,
    horizon: int = 30,
    verbose: bool = True,
) -> CVResult:
    """
    Expanding-window cross-validation for SARIMAX.

    Splits the data into k folds, each with a test window of `horizon` days.
    Training expands forward with each fold.
    """
    n = len(endog)
    fold_rmses: list[float] = []

    for i in range(k):
        test_end = n - (k - 1 - i) * horizon
        test_start = test_end - horizon
        train_end = test_start

        y_train = endog.iloc[:train_end]
        x_train = exog.iloc[:train_end]
        y_test = endog.iloc[test_start:test_end]
        x_test = exog.iloc[test_start:test_end]

        model = SARIMAX(
            y_train,
            exog=x_train,
            order=params.order,
            seasonal_order=params.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=horizon, exog=x_test)
        rmse_val = root_mean_squared_error(y_test, forecast)
        fold_rmses.append(rmse_val)

        if verbose:
            print(
                f"Fold {i + 1}: train={train_end}, "
                f"test=[{test_start}:{test_end}], RMSE={rmse_val:.4f}"
            )

    avg = float(np.mean(fold_rmses))
    if verbose:
        print(f"\nAverage RMSE across {k} folds: {avg:.4f}")

    return CVResult(avg_rmse=avg, fold_rmses=fold_rmses, params=params)


# ---------------------------------------------------------------------------
# SARIMAX
# ---------------------------------------------------------------------------


def select_sarimax(
    endog: pd.Series,
    exog: pd.DataFrame,
    candidates: list[SARIMAXParams],
    k: int = 5,
    horizon: int = 30,
    verbose: bool = True,
) -> GridSearchResult:
    """
    Grid-search over SARIMAX parameter candidates using rolling CV.
    Returns a GridSearchResult with the best model fitted on full data.
    """
    best_cv: CVResult | None = None
    all_results: list[CVResult] = []
    failed = 0

    for params in candidates:
        if verbose:
            print(f"\n--- {params} ---")
        try:
            cv = rolling_cv(endog, exog, params, k=k, horizon=horizon, verbose=verbose)
            all_results.append(cv)
            if best_cv is None or cv.avg_rmse < best_cv.avg_rmse:
                best_cv = cv
        except Exception as e:
            failed += 1
            if verbose:
                print(f"Failed: {e}")

    if best_cv is None:
        raise ValueError("No valid SARIMAX candidate was found.")

    if verbose:
        print(f"\nBest: {best_cv.params}, RMSE={best_cv.avg_rmse:.4f}")

    fitted = fit_sarimax(endog, exog, best_cv.params)
    fitted.cv_result = best_cv

    return GridSearchResult(best=fitted, all_results=all_results, failed=failed)


def fit_sarimax(
    endog: pd.Series,
    exog: pd.DataFrame,
    params: SARIMAXParams,
) -> SARIMAXResult:
    """Fit a SARIMAX model on the full data."""
    model = SARIMAX(
        endog,
        exog=exog,
        order=params.order,
        seasonal_order=params.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    return SARIMAXResult(fitted=fitted, params=params)


def forecast_sarimax(
    result: SARIMAXResultsWrapper,
    steps: int,
    future_exog: pd.DataFrame,
) -> pd.Series:
    """Generate forecasts from a fitted SARIMAX model."""
    return result.fitted.forecast(steps=steps, exog=future_exog)


# ---------------------------------------------------------------------------
# Exogenous variable helpers
# ---------------------------------------------------------------------------


def make_future_exog(
    df: pd.DataFrame,
    steps: int = 30,
    exog_cols: list[str] | None = None,
    start: str | None = None,
) -> pd.DataFrame:
    """
    Build a future exogenous DataFrame for forecasting.

    Fills numeric columns with their historical mean and binary columns with 0.
    Override specific column values by editing the returned DataFrame.
    """
    if exog_cols is None:
        exog_cols = [c for c in df.columns if c != "sales"]

    last_date = (
        df.index.max() if start is None else pd.Timestamp(start) - pd.Timedelta(days=1)
    )
    future_idx = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=steps, freq="D"
    )

    data: dict[str, list[float]] = {}
    for col in exog_cols:
        is_binary = df[col].dropna().isin([0, 1]).all()
        data[col] = [0.0 if is_binary else float(df[col].mean())] * steps

    return pd.DataFrame(data, index=future_idx)


# ---------------------------------------------------------------------------
# Prophet
# ---------------------------------------------------------------------------


def fit_prophet(
    df: pd.DataFrame,
    target_col: str = "sales",
    regressor_cols: list[str] | None = None,
    holiday_col: str | None = "holiday",
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
) -> Prophet:
    """
    Fit a Prophet model.

    Parameters
    ----------
    df : DataFrame with DatetimeIndex
    target_col : column to forecast
    regressor_cols : columns to add as extra regressors
    holiday_col : binary column indicating holiday periods
    """
    df_p = df.reset_index().rename(
        columns={df.index.name or "date": "ds", target_col: "y"}
    )

    holidays_df = None
    if holiday_col and holiday_col in df_p.columns:
        h = df_p[df_p[holiday_col] == 1][["ds"]].copy()
        h["holiday"] = "promo_holiday"
        h["lower_window"] = 0
        h["upper_window"] = 0
        holidays_df = h

    m = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=False,
        holidays=holidays_df,
    )

    fit_cols = ["ds", "y"]
    if regressor_cols:
        for col in regressor_cols:
            m.add_regressor(col)
            fit_cols.append(col)

    m.fit(df_p[fit_cols])
    return m


def forecast_prophet(
    model: Prophet,
    df: pd.DataFrame,
    periods: int = 30,
    regressor_defaults: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Generate Prophet forecast for future periods.

    Parameters
    ----------
    model : fitted Prophet model
    df : original DataFrame (used for historical regressor values)
    periods : number of days to forecast
    regressor_defaults : default values for regressors in the future period;
                         if not provided, uses the historical mean
    """
    df_p = df.reset_index().rename(columns={df.index.name or "date": "ds"})
    future = model.make_future_dataframe(periods=periods)

    regressor_names = list(model.extra_regressors.keys())

    for col in regressor_names:
        if col in df_p.columns:
            future = future.merge(df_p[["ds", col]], on="ds", how="left")
        default = (regressor_defaults or {}).get(
            col, float(df[col].mean()) if col in df.columns else 0.0
        )
        future[col] = future[col].fillna(default)

    forecast = model.predict(future)
    return forecast.tail(periods)
