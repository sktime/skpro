"""Probabilistic regression with constant training variance."""

__author__ = ["NandiniDhanrale"]
__all__ = ["ConstantVarianceRegressor"]

import numpy as np
import pandas as pd
from sklearn import clone

from skpro.distributions.laplace import Laplace
from skpro.distributions.normal import Normal
from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class ConstantVarianceRegressor(BaseProbaRegressor):
    """Probabilistic regressor with mean from a deterministic regressor.

    This regressor wraps a deterministic tabular regressor and uses its point
    predictions as the predictive mean. The predictive variance is taken to be
    constant across all instances and equal to the sample variance of the training
    targets seen in ``fit``.

    This implements a simple roadmap baseline: deterministic mean prediction with
    uncertainty estimated from the training sample alone.

    Parameters
    ----------
    estimator : sklearn regressor
        Deterministic regressor used to predict the mean.
    distribution : {"Normal", "Laplace"}, default="Normal"
        Distribution family used for ``predict_proba``.
        ``"Normal"`` uses the training standard deviation as ``sigma``.
        ``"Laplace"`` uses a scale chosen so that the predictive variance matches
        the training sample variance.
    """

    _tags = {
        "authors": ["NandiniDhanrale"],
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, estimator, distribution="Normal"):
        self.estimator = estimator
        self.distribution = distribution
        super().__init__()

    def _get_distribution_name(self):
        distribution = self.distribution.lower()
        valid_distributions = {"normal": "Normal", "laplace": "Laplace"}

        if distribution not in valid_distributions:
            raise ValueError(
                "distribution must be one of {'Normal', 'Laplace'}, "
                f"but found {self.distribution!r}"
            )

        return valid_distributions[distribution]

    def _fit(self, X, y):
        """Fit regressor to training data."""
        X = prep_skl_df(X, copy_df=True)
        y = y.copy()

        self._get_distribution_name()
        self._y_cols = y.columns
        self._y_var = y.var(axis=0, ddof=0)

        y_inner = y.iloc[:, 0] if y.shape[1] == 1 else y

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y_inner)
        return self

    def _predict(self, X):
        """Predict labels for data from features."""
        X = prep_skl_df(X, copy_df=True)
        y_pred = self.estimator_.predict(X)

        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.to_frame()
        elif not isinstance(y_pred, pd.DataFrame):
            y_pred = pd.DataFrame(y_pred, index=X.index, columns=self._y_cols)

        return y_pred

    def _predict_var(self, X):
        """Compute/return variance predictions."""
        X = prep_skl_df(X, copy_df=True)
        var_arr = np.tile(self._y_var.to_numpy(), (len(X), 1))
        return pd.DataFrame(var_arr, index=X.index, columns=self._y_cols)

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features."""
        X = prep_skl_df(X, copy_df=True)
        y_pred = self._predict(X)
        pred_var = self._predict_var(X)
        distribution = self._get_distribution_name()

        if distribution == "Normal":
            scale = np.sqrt(pred_var.to_numpy())
            return Normal(
                mu=y_pred.to_numpy(),
                sigma=scale,
                index=X.index,
                columns=self._y_cols,
            )

        scale = np.sqrt(pred_var.to_numpy() / 2)
        return Laplace(
            mu=y_pred.to_numpy(),
            scale=scale,
            index=X.index,
            columns=self._y_cols,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        params1 = {"estimator": LinearRegression()}
        params2 = {
            "estimator": RandomForestRegressor(n_estimators=3, random_state=0),
            "distribution": "Laplace",
        }

        return [params1, params2]
