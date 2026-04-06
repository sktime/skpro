import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from skpro.distributions.normal import Normal
from skpro.regression.ensemble import (
    ProbabilisticBoostingRegressor,
    ProbabilisticStackingRegressor,
)
from skpro.regression.residual import ResidualDouble


def _interval(self, alpha):
    alpha = np.asarray(alpha)
    lower = self.ppf((1 - alpha) / 2)
    upper = self.ppf(1 - (1 - alpha) / 2)
    return lower, upper


Normal.interval = _interval


@pytest.fixture
def simple_data():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    y = pd.DataFrame({"target": [1.1, 2.2, 3.3]})
    return X, y


def test_probabilistic_stacking_regressor_basic(simple_data):
    X, y = simple_data
    est1 = ResidualDouble(LinearRegression())
    est2 = ResidualDouble(LinearRegression())
    stack = ProbabilisticStackingRegressor(estimators=[("est1", est1), ("est2", est2)])
    stack.fit(X, y)
    y_pred = stack.predict_proba(X)
    assert hasattr(y_pred, "mean"), "Stacking output should have mean method"
    assert y_pred.mean().shape[0] == X.shape[0]


def test_probabilistic_stacking_regressor_weights(simple_data):
    X, y = simple_data
    est1 = ResidualDouble(LinearRegression())
    est2 = ResidualDouble(LinearRegression())
    weights = [0.8, 0.2]
    stack = ProbabilisticStackingRegressor(
        estimators=[("est1", est1), ("est2", est2)], weights=weights
    )
    stack.fit(X, y)
    y_pred = stack.predict_proba(X)
    assert hasattr(y_pred, "mean")
    assert y_pred.mean().shape[0] == X.shape[0]


def test_probabilistic_stacking_regressor_error_on_mismatch(simple_data):
    est1 = ResidualDouble(LinearRegression())
    est2 = ResidualDouble(LinearRegression())
    weights = [0.5]  # wrong length
    with pytest.raises(ValueError, match="weights must have the same length"):
        ProbabilisticStackingRegressor(
            estimators=[("est1", est1), ("est2", est2)], weights=weights
        )


def test_probabilistic_boosting_regressor_basic(simple_data):
    X, y = simple_data
    est1 = ResidualDouble(LinearRegression())
    boost = ProbabilisticBoostingRegressor(base_estimator=est1, n_estimators=2)
    boost.fit(X, y)
    y_pred = boost.predict_proba(X)
    assert hasattr(y_pred, "mean"), "Boosting output should have mean method"
    assert y_pred.mean().shape[0] == X.shape[0]


def test_probabilistic_boosting_regressor_learning_rate(simple_data):
    X, y = simple_data
    est1 = ResidualDouble(LinearRegression())
    boost = ProbabilisticBoostingRegressor(
        base_estimator=est1, n_estimators=3, learning_rate=0.5
    )
    boost.fit(X, y)
    y_pred = boost.predict_proba(X)
    assert hasattr(y_pred, "mean")
    assert y_pred.mean().shape[0] == X.shape[0]


def test_probabilistic_stacking_regressor_quantile_output(simple_data):
    X, y = simple_data
    est1 = ResidualDouble(LinearRegression())
    est2 = ResidualDouble(LinearRegression())
    stack = ProbabilisticStackingRegressor(estimators=[("est1", est1), ("est2", est2)])
    stack.fit(X, y)
    y_pred = stack.predict_proba(X)
    if hasattr(y_pred, "quantile"):
        q = y_pred.quantile(0.5)
        assert q.shape[0] == X.shape[0]


def test_probabilistic_boosting_regressor_quantile_output(simple_data):
    X, y = simple_data
    est1 = ResidualDouble(LinearRegression())
    boost = ProbabilisticBoostingRegressor(base_estimator=est1, n_estimators=2)
    boost.fit(X, y)
    y_pred = boost.predict_proba(X)
    if hasattr(y_pred, "quantile"):
        q = y_pred.quantile(0.5)
        assert q.shape[0] == X.shape[0]
