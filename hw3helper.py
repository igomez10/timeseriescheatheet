import warnings

import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA, ARIMAResultsWrapper
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")


def evaluate_arima_model(
    X: np.ndarray, train_ratio: float, arima_order: tuple, trend: str | list[int]
) -> float:

    # split into train and test sets
    train_size = int(len(X) * train_ratio)
    train = X[0:train_size]
    test = X[train_size:]

    # fit model
    model = ARIMA(train, order=arima_order, trend=trend)
    model_fit = model.fit()
    predictions = model_fit.forecast(len(test))

    # calculate out of sample error
    rmse = np.sqrt(mean_squared_error(test, predictions))
    return rmse


def evaluate_arima_models(
    dataset: np.ndarray,
    train_ratio: float,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    trend_values: list[str | list[int]],
) -> tuple[tuple[int, int, int] | None, str | list[int] | None]:
    result = []
    best_rmse, best_cfg, best_trend = float("inf"), None, None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for trend in trend_values:
                    try:
                        order = (p, d, q)
                        rmse: float = evaluate_arima_model(
                            dataset, train_ratio, order, trend
                        )
                        result.append((order, rmse))
                        if rmse < best_rmse:
                            best_rmse = rmse
                            best_cfg = order
                            best_trend = trend
                    except Exception:
                        continue

    return best_cfg, best_trend


class find_best_arima_response:
    def __init__(
        self,
        model: ARIMAResultsWrapper,
        rmse: float,
        p: int,
        d: int,
        q: int,
        trend: str | list[int],
    ):
        self.trend = trend
        self.model = model
        self.rmse = rmse
        self.p = p
        self.d = d
        self.q = q

    def __str__(self) -> str:
        return f"ARIMA({self.p}, {self.d}, {self.q}) with trend {self.trend} has RMSE: {self.rmse}"

    def train_model(self, train_data: np.ndarray) -> ARIMAResultsWrapper:
        """
        Train a new ARIMA model using the p, d, q and trend from the best model found.
        """
        model = ARIMA(train_data, order=(self.p, self.d, self.q), trend=self.trend)
        return model.fit()


def find_best_arima(
    x_train: np.ndarray,
    x_test: np.ndarray,
    p_values: list[int],
    d_values: list[int],
    q_values: list[int],
    trend_values: list[str | list[int]],
) -> find_best_arima_response:
    res = None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                for trend in trend_values:
                    try:
                        # build model
                        order = (p, d, q)
                        model = ARIMA(x_train, order=order, trend=trend)
                        # fit model
                        model_fit = model.fit()
                        predictions = model_fit.forecast(len(x_test))
                        # evaluate forecasts
                        rmse = np.sqrt(mean_squared_error(x_test, predictions))
                        if res is None or rmse < res.rmse:
                            res = find_best_arima_response(
                                model=model_fit,
                                rmse=rmse,
                                p=p,
                                d=d,
                                q=q,
                                trend=trend,
                            )
                    except Exception:
                        continue
    if res is None:
        raise ValueError("No valid ARIMA model found")

    return res


class find_best_sarima_response:
    def __init__(
        self,
        model: ARIMAResultsWrapper,
        rmse: float,
        p: int,
        d: int,
        q: int,
        seasonal_P: int,
        seasonal_D: int,
        seasonal_Q: int,
        seasonal_period: int,
        trend: str | list[int],
    ):
        self.trend = trend
        self.model = model
        self.rmse = rmse
        self.p = p
        self.d = d
        self.q = q
        self.seasonal_P = seasonal_P
        self.seasonal_D = seasonal_D
        self.seasonal_Q = seasonal_Q
        self.seasonal_period = seasonal_period

    def __str__(self) -> str:
        return f"SARIMA({self.p}, {self.d}, {self.q}) x ({self.seasonal_P}, {self.seasonal_D}, {self.seasonal_Q}, {self.seasonal_period}) with trend {self.trend} has RMSE: {self.rmse}"

    def train_model(
        self, train_data: np.ndarray, seasonal_period: int
    ) -> ARIMAResultsWrapper:
        """
        Train a new SARIMA model using the p, d, q and trend from the best model found.
        """
        model = SARIMAX(
            train_data,
            order=(self.p, self.d, self.q),
            seasonal_order=(
                self.seasonal_P,
                self.seasonal_D,
                self.seasonal_Q,
                seasonal_period,
            ),
            trend=self.trend,
        )
        return model.fit()


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
    for p in p_values:
        for d in d_values:
            for q in q_values:
                for P in P_values:
                    for D in D_values:
                        for Q in Q_values:
                            for trend in trend_values:
                                try:
                                    # build model
                                    order = (p, d, q)
                                    seasonal_order = (P, D, Q, seasonal_period)
                                    model = SARIMAX(
                                        x_train,
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        trend=trend,
                                    )
                                    # fit model
                                    model_fit = model.fit()
                                    predictions = model_fit.forecast(len(x_test))
                                    # evaluate forecasts
                                    rmse = np.sqrt(
                                        mean_squared_error(x_test, predictions)
                                    )
                                    if res is None or rmse < res.rmse:
                                        res = find_best_sarima_response(
                                            model=model_fit,
                                            rmse=rmse,
                                            p=p,
                                            d=d,
                                            q=q,
                                            seasonal_P=P,
                                            seasonal_D=D,
                                            seasonal_Q=Q,
                                            seasonal_period=seasonal_period,
                                            trend=trend,
                                        )
                                except Exception as e:
                                    print(
                                        f"Error fitting SARIMA({p}, {d}, {q}) x ({P}, {D}, {Q}, {seasonal_period}) with trend {trend}: {e}"
                                    )
                                    continue
    if res is None:
        raise ValueError("No valid SARIMA model found")

    return res
