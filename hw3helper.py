import warnings
from dataclasses import dataclass
from itertools import product

import numpy as np
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResultsWrapper

warnings.filterwarnings("ignore")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate_arima_model(
    X: np.ndarray, train_ratio: float, arima_order: tuple, trend: str | list[int]
) -> float:
    train_size = int(len(X) * train_ratio)
    train, test = X[:train_size], X[train_size:]
    predictions = ARIMA(train, order=arima_order, trend=trend).fit().forecast(len(test))
    return _rmse(test, predictions)


def evaluate_arima_models(
    dataset: np.ndarray,
    train_ratio: float,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    trend_values: list[str | list[int]],
) -> tuple[tuple[int, int, int] | None, str | list[int] | None]:
    best_rmse, best_cfg, best_trend = float("inf"), None, None
    for p, d, q, trend in product(p_values, d_values, q_values, trend_values):
        try:
            rmse = evaluate_arima_model(dataset, train_ratio, (p, d, q), trend)
            if rmse < best_rmse:
                best_rmse, best_cfg, best_trend = rmse, (p, d, q), trend
        except Exception:
            continue
    return best_cfg, best_trend


@dataclass
class find_best_arima_response:
    model: ARIMAResultsWrapper
    rmse: float
    p: int
    d: int
    q: int
    trend: str | list[int]

    def __str__(self) -> str:
        return f"ARIMA({self.p}, {self.d}, {self.q}) with trend {self.trend} has RMSE: {self.rmse}"

    def train_model(self, train_data: np.ndarray) -> ARIMAResultsWrapper:
        return ARIMA(train_data, order=(self.p, self.d, self.q), trend=self.trend).fit()


def find_best_arima(
    x_train: np.ndarray,
    x_test: np.ndarray,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    trend_values: list[str | list[int]] | None = None,
) -> find_best_arima_response:
    if trend_values is None:
        trend_values = []
    res = None
    for p, d, q, trend in product(p_values, d_values, q_values, trend_values):
        try:
            model_fit = ARIMA(x_train, order=(p, d, q), trend=trend).fit()
            rmse = _rmse(x_test, model_fit.forecast(len(x_test)))
            if res is None or rmse < res.rmse:
                res = find_best_arima_response(model=model_fit, rmse=rmse, p=p, d=d, q=q, trend=trend)
        except Exception:
            continue
    if res is None:
        raise ValueError("No valid ARIMA model found")
    return res


@dataclass
class find_best_sarima_response:
    model: SARIMAXResultsWrapper
    rmse: float
    p: int
    d: int
    q: int
    seasonal_P: int
    seasonal_D: int
    seasonal_Q: int
    seasonal_period: int
    trend: str | list[int]

    def __str__(self) -> str:
        return (
            f"SARIMA({self.p}, {self.d}, {self.q}) x "
            f"({self.seasonal_P}, {self.seasonal_D}, {self.seasonal_Q}, {self.seasonal_period}) "
            f"with trend {self.trend} has RMSE: {self.rmse}"
        )

    def train_model(self, train_data: np.ndarray, seasonal_period: int) -> SARIMAXResultsWrapper:
        return SARIMAX(
            train_data,
            order=(self.p, self.d, self.q),
            seasonal_order=(self.seasonal_P, self.seasonal_D, self.seasonal_Q, seasonal_period),
            trend=self.trend,
        ).fit()


def find_best_sarima(
    x_train: np.ndarray,
    x_test: np.ndarray,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    P_values: list[int],
    D_values: list[int],
    Q_values: list[int],
    seasonal_period: int,
    trend_values: list[str | list[int]],
) -> find_best_sarima_response:
    res = None
    for p, d, q, P, D, Q, trend in product(
        p_values, d_values, q_values, P_values, D_values, Q_values, trend_values
    ):
        try:
            seasonal_order = (P, D, Q, seasonal_period)
            model_fit = SARIMAX(
                x_train, order=(p, d, q), seasonal_order=seasonal_order, trend=trend
            ).fit()
            rmse = _rmse(x_test, model_fit.forecast(len(x_test)))
            if res is None or rmse < res.rmse:
                res = find_best_sarima_response(
                    model=model_fit, rmse=rmse, p=p, d=d, q=q,
                    seasonal_P=P, seasonal_D=D, seasonal_Q=Q,
                    seasonal_period=seasonal_period, trend=trend,
                )
        except Exception as e:
            print(f"Error fitting SARIMA({p}, {d}, {q}) x ({P}, {D}, {Q}, {seasonal_period}) with trend {trend}: {e}")
            continue
    if res is None:
        raise ValueError("No valid SARIMA model found")
    return res


@dataclass
class find_best_sarimax_response:
    model: SARIMAXResultsWrapper
    rmse: float
    p: int
    d: int
    q: int
    seasonal_P: int
    seasonal_D: int
    seasonal_Q: int
    seasonal_period: int
    trend: str | list[int]

    def __str__(self) -> str:
        return (
            f"SARIMAX({self.p}, {self.d}, {self.q}) x "
            f"({self.seasonal_P}, {self.seasonal_D}, {self.seasonal_Q}, {self.seasonal_period}) "
            f"with trend {self.trend} has RMSE: {self.rmse}"
        )

    def train_model(
        self, train_data: np.ndarray, exogenous: np.ndarray, seasonal_period: int
    ) -> SARIMAXResultsWrapper:
        return SARIMAX(
            train_data,
            exog=exogenous,
            order=(self.p, self.d, self.q),
            seasonal_order=(self.seasonal_P, self.seasonal_D, self.seasonal_Q, seasonal_period),
            trend=self.trend,
        ).fit()


def find_best_sarimax(
    x_train: np.ndarray,
    exogenous_train: np.ndarray,
    x_test: np.ndarray,
    exogenous_test: np.ndarray,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    P_values: list[int],
    D_values: list[int],
    Q_values: list[int],
    seasonal_period: int,
    trend_values: list[str | list[int]],
) -> find_best_sarimax_response:
    res = None
    for p, d, q, P, D, Q, trend in product(
        p_values, d_values, q_values, P_values, D_values, Q_values, trend_values
    ):
        try:
            seasonal_order = (P, D, Q, seasonal_period)
            model_fit = SARIMAX(
                x_train, exog=exogenous_train, order=(p, d, q),
                seasonal_order=seasonal_order, trend=trend,
            ).fit()
            rmse = _rmse(x_test, model_fit.forecast(steps=len(x_test), exog=exogenous_test))
            if res is None or rmse < res.rmse:
                res = find_best_sarimax_response(
                    model=model_fit, rmse=rmse, p=p, d=d, q=q,
                    seasonal_P=P, seasonal_D=D, seasonal_Q=Q,
                    seasonal_period=seasonal_period, trend=trend,
                )
        except Exception as e:
            print(f"Error fitting SARIMAX({p}, {d}, {q}) x ({P}, {D}, {Q}, {seasonal_period}) with trend {trend}: {e}")
            continue
    if res is None:
        raise ValueError("No valid SARIMAX model found")
    return res
