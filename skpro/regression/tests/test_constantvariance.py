"""Tests for the ConstantVarianceRegressor."""

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from skpro.distributions.laplace import Laplace
from skpro.distributions.normal import Normal
from skpro.regression.constantvariance import ConstantVarianceRegressor


@pytest.mark.parametrize(
    "distribution, expected_distr",
    [("Normal", Normal), ("Laplace", Laplace)],
)
def test_constant_variance_regressor_predictive_distribution(
    distribution, expected_distr
):
    """Test predictive mean and variance for the constant-variance wrapper."""
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:80]
    y = y.iloc[:80]
    X_train, X_test, y_train, _ = train_test_split(X, y, random_state=0)

    reg = ConstantVarianceRegressor(
        estimator=LinearRegression(),
        distribution=distribution,
    )
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    y_var = reg.predict_var(X_test)
    y_proba = reg.predict_proba(X_test)

    expected_var = pd.DataFrame(
        y_train.var(axis=0, ddof=0).to_numpy().reshape(1, -1).repeat(len(X_test), axis=0),
        index=X_test.index,
        columns=y_train.columns,
    )

    assert isinstance(y_proba, expected_distr)
    pdt.assert_frame_equal(y_proba.mean(), y_pred)
    pdt.assert_frame_equal(y_var, expected_var)
    pdt.assert_frame_equal(y_proba.var(), expected_var)


def test_constant_variance_regressor_invalid_distribution():
    """Test invalid distribution name raises a clear error."""
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import LinearRegression

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)

    reg = ConstantVarianceRegressor(
        estimator=LinearRegression(),
        distribution="Cauchy",
    )

    with pytest.raises(ValueError, match="distribution must be one of"):
        reg.fit(X.iloc[:20], y.iloc[:20])


@pytest.mark.parametrize(
    "distribution, expected_distr",
    [("Normal", Normal), ("Laplace", Laplace)],
)
def test_constant_variance_regressor_constant_target_is_numerically_stable(
    distribution, expected_distr
):
    """Constant targets should still yield a valid predictive distribution."""
    from sklearn.linear_model import LinearRegression

    X = pd.DataFrame({"x": [0, 1, 2, 3]})
    y = pd.DataFrame({"y": [1.0, 1.0, 1.0, 1.0]})

    reg = ConstantVarianceRegressor(
        estimator=LinearRegression(),
        distribution=distribution,
    )
    reg.fit(X, y)

    y_var = reg.predict_var(X)
    y_proba = reg.predict_proba(X)

    assert isinstance(y_proba, expected_distr)
    assert (y_var.to_numpy() > 0).all()
    assert np.isfinite(y_proba.pdf(y_proba.mean()).to_numpy()).all()
    pdt.assert_frame_equal(y_proba.var(), y_var)
