from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm


@dataclass
class AutoARIMAResult:
    """Result from fitting an auto_arima model."""

    model: pm.ARIMA
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float

    def __str__(self) -> str:
        return f"ARIMA{self.order}x{self.seasonal_order} AIC={self.aic:.2f}"


@dataclass
class ForecastResult:
    """Forecast values with optional confidence intervals."""

    values: np.ndarray
    conf_int: np.ndarray | None = None


@dataclass
class EvalResult:
    """Evaluation metrics for a forecast."""

    rmse: float
    mae: float


def fit_auto_arima(
    y: pd.Series | np.ndarray,
    seasonal: bool = True,
    m: int = 12,
    d: int | None = None,
    D: int | None = None,
    start_p: int = 0,
    max_p: int = 5,
    start_q: int = 0,
    max_q: int = 5,
    start_P: int = 0,
    max_P: int = 2,
    start_Q: int = 0,
    max_Q: int = 2,
    information_criterion: str = "aic",
    stepwise: bool = True,
    trace: bool = False,
) -> AutoARIMAResult:
    """
    Fit an auto_arima model, automatically selecting (p,d,q)x(P,D,Q,m).

    Parameters
    ----------
    y : training series
    seasonal : whether to fit seasonal components
    m : seasonal period (e.g. 12 for monthly, 7 for daily-weekly)
    d, D : differencing orders; None lets auto_arima decide via tests
    start_p, start_q, start_P, start_Q : lower bounds for grid search
    max_p, max_q, max_P, max_Q : upper bounds for grid search
    information_criterion : 'aic' or 'bic'
    stepwise : use stepwise search (faster) or exhaustive
    trace : print models evaluated during search
    """
    model = pm.auto_arima(
        y,
        seasonal=seasonal,
        m=m,
        d=d,
        D=D,
        start_p=start_p,
        max_p=max_p,
        start_q=start_q,
        max_q=max_q,
        start_P=start_P,
        max_P=max_P,
        start_Q=start_Q,
        max_Q=max_Q,
        stepwise=stepwise,
        trace=trace,
        error_action="ignore",
        suppress_warnings=True,
        information_criterion=information_criterion,
    )
    return AutoARIMAResult(
        model=model,
        order=model.order,
        seasonal_order=model.seasonal_order,
        aic=model.aic(),
    )


def forecast(
    result: AutoARIMAResult,
    n_periods: int,
    return_conf_int: bool = True,
) -> ForecastResult:
    """Generate n-step ahead forecast from a fitted AutoARIMAResult."""
    if return_conf_int:
        vals, ci = result.model.predict(
            n_periods=n_periods, return_conf_int=True
        )
        return ForecastResult(values=vals, conf_int=ci)
    vals = result.model.predict(n_periods=n_periods)
    return ForecastResult(values=vals, conf_int=None)


def evaluate_forecast(actual: np.ndarray | pd.Series, predicted: np.ndarray) -> EvalResult:
    """Compute RMSE and MAE between actual and predicted values."""
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))
    return EvalResult(rmse=rmse, mae=mae)


def update_model(
    result: AutoARIMAResult,
    new_obs: pd.Series | np.ndarray,
) -> None:
    """Update the model in-place with new observations (online learning)."""
    result.model.update(new_obs)


def plot_forecast(
    train: pd.Series,
    test: pd.Series | None,
    fc: ForecastResult,
    fc_index: pd.Index | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot train, optional test, and forecast with confidence band.

    Parameters
    ----------
    train : training series
    test : test series (optional, for comparison)
    fc : ForecastResult from forecast()
    fc_index : DatetimeIndex for the forecast; defaults to RangeIndex
    title : plot title
    ax : matplotlib Axes to draw on; creates new figure if None
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.plot(train.index, train, label="Train")

    if fc_index is None:
        fc_index = pd.RangeIndex(len(train), len(train) + len(fc.values))

    if test is not None:
        ax.plot(test.index, test, label="Test", color="gray")

    ax.plot(fc_index, fc.values, label="Forecast", color="red")

    if fc.conf_int is not None:
        ax.fill_between(
            fc_index, fc.conf_int[:, 0], fc.conf_int[:, 1],
            color="red", alpha=0.15,
        )

    if title:
        ax.set_title(title)
    ax.legend()
    return ax


def train_test_split(
    y: pd.Series, n_test: int
) -> tuple[pd.Series, pd.Series]:
    """Split a series into train and test by holding out the last n_test observations."""
    return y[:-n_test], y[-n_test:]
