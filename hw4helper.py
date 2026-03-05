from dataclasses import dataclass
from itertools import product

import numpy as np
import sklearn.metrics as skmetrics
from statsmodels.tsa.arima.model import ARIMA


@dataclass
class CVResult:
    order: tuple
    step_rmses: list[float]  # RMSE per forecast horizon
    avg_rmse: float


def evaluate_arima_cv(
    dataset,
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
    initial_train_size = len(dataset) - n_folds * forecast_horizon
    results = []

    for order in product(p_values, d_values, q_values):
        fold_step_mses = [[] for _ in range(forecast_horizon)]
        fold_rmses = []

        for k in range(n_folds):
            split = initial_train_size + k * forecast_horizon
            train = dataset[:split]
            test = dataset[split : split + forecast_horizon]

            model = ARIMA(train, order=order).fit()
            forecast = model.forecast(forecast_horizon)

            for step in range(forecast_horizon):
                mse = skmetrics.mean_squared_error(
                    test[step : step + 1], forecast[step : step + 1]
                )
                fold_step_mses[step].append(mse)

            fold_rmses.append(np.sqrt(skmetrics.mean_squared_error(test, forecast)))

        step_rmses = [np.sqrt(np.mean(mses)) for mses in fold_step_mses]
        avg_rmse = np.mean(fold_rmses)
        results.append(CVResult(order, step_rmses, avg_rmse))

    return results


def print_best_models(results: list[CVResult]) -> None:
    """Print the best model for each forecast step and overall."""
    if not results:
        return

    forecast_horizon = len(results[0].step_rmses)

    for step in range(forecast_horizon):
        best = min(results, key=lambda r, s=step: r.step_rmses[s])
        print(f"Best Step {step + 1}: {best.order}  RMSE={best.step_rmses[step]:.4f}")

    best_avg = min(results, key=lambda r: r.avg_rmse)
    print(f"Best Avg:    {best_avg.order}  RMSE={best_avg.avg_rmse:.4f}")
