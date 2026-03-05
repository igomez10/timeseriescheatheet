from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class CVResult:
    order: tuple
    step_rmses: list[float]  # RMSE per forecast horizon
    avg_rmse: float


def _to_series(dataset: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(dataset, pd.DataFrame):
        return dataset.iloc[:, 0]
    return dataset


def _direct_forecast(
    train: pd.Series, test: pd.Series, order: tuple, forecast_horizon: int
) -> list[float]:
    forecast = ARIMA(train, order=order).fit().forecast(forecast_horizon)
    return list((test.values - forecast.values) ** 2)


def _recursive_forecast(
    train: pd.Series, test: pd.Series, order: tuple, forecast_horizon: int
) -> list[float]:
    train = train.copy()
    freq = getattr(train.index, "freq", None)
    step_mses = []
    for step in range(forecast_horizon):
        prediction = ARIMA(train, order=order).fit().forecast(1)
        step_mses.append(float((test.iloc[step] - prediction.iloc[0]) ** 2))
        train = pd.concat([train, prediction.rename(train.name)])
        if freq is not None:
            train.index.freq = freq
    return step_mses


def _evaluate_orders(
    dataset: pd.Series,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    forecast_horizon: int,
    n_folds: int,
    forecast_fn,
) -> list[CVResult]:
    initial_train_size = len(dataset) - n_folds * forecast_horizon
    results = []

    for order in product(p_values, d_values, q_values):
        fold_step_mses = [[] for _ in range(forecast_horizon)]
        fold_rmses = []

        for k in range(n_folds):
            split = initial_train_size + k * forecast_horizon
            train = dataset[:split]
            test = dataset[split : split + forecast_horizon]

            step_mses = forecast_fn(train, test, order, forecast_horizon)

            for step, mse in enumerate(step_mses):
                fold_step_mses[step].append(mse)
            fold_rmses.append(np.sqrt(np.mean(step_mses)))

        step_rmses = [np.sqrt(np.mean(mses)) for mses in fold_step_mses]
        avg_rmse = np.mean(fold_rmses)
        results.append(CVResult(order, step_rmses, avg_rmse))

    return results


def evaluate_arima_cv(
    dataset: pd.Series,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    forecast_horizon: int,
    n_folds: int,
) -> list[CVResult]:
    """
    Evaluate ARIMA models over a grid of (p, d, q) orders using
    rolling-origin cross-validation.

    Each fold expands the training window by `forecast_horizon` steps,
    then scores the model on per-step RMSE and overall RMSE.
    """
    return _evaluate_orders(
        _to_series(dataset), p_values, d_values, q_values,
        forecast_horizon, n_folds, _direct_forecast,
    )


def evaluate_arima_recursive_cv(
    dataset: pd.Series,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    forecast_horizon: int,
    n_folds: int,
) -> list[CVResult]:
    """
    Evaluate ARIMA models using recursive multi-step forecasting
    with rolling-origin cross-validation.

    At each forecast step, the model is refit with the previous
    step's prediction appended to the training set.
    """
    return _evaluate_orders(
        _to_series(dataset), p_values, d_values, q_values,
        forecast_horizon, n_folds, _recursive_forecast,
    )


def print_best_models(results: list[CVResult]) -> None:
    """Print the best model for each forecast step and overall."""
    if not results:
        return

    forecast_horizon = len(results[0].step_rmses)

    for step in range(forecast_horizon):
        best = min(results, key=lambda r: r.step_rmses[step])
        print(f"Best Step {step + 1}: {best.order}  RMSE={best.step_rmses[step]:.4f}")

    best_avg = min(results, key=lambda r: r.avg_rmse)
    print(f"Best Avg:    {best_avg.order}  RMSE={best_avg.avg_rmse:.4f}")
