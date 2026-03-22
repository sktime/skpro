"""Reducing Interval Regressor: produces intervals that shrink as more data is seen.

This regressor demonstrates two approaches for interval prediction:
- Using mean and standard deviation (Normal assumption)
- Using quantile regression (empirical quantiles)

Implements both _predict_interval and _predict_quantiles for demonstration.
"""

import numpy as np
import pandas as pd

from skpro.regression.base import BaseProbaRegressor


class ReducingIntervalRegressor(BaseProbaRegressor):
    """Probabilistic regressor with reducing intervals as n increases.

    Parameters
    ----------
    method : str, default="mean_sd"
        "mean_sd" for mean/sd-based intervals, "quantile" for empirical quantiles.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from skpro.regression.reducing_interval import ReducingIntervalRegressor
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> y = pd.DataFrame(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> reg = ReducingIntervalRegressor(method="mean_sd")
    >>> reg.fit(X_train, y_train)
    ReducingIntervalRegressor(...)
    >>> y_pred = reg.predict(X_test)
    >>> intervals = reg.predict_interval(X_test, coverage=[0.9])
    >>> quantiles = reg.predict_quantiles(X_test, alpha=[0.05, 0.5, 0.95])
    """

    _tags = {
        "authors": ["arnavk23"],
        "capability:multioutput": False,
        "capability:missing": True,
    }

    def __init__(self, method="mean_sd"):
        self.method = method
        super().__init__()

    def _fit(self, X, y, C=None):
        # Store training data for empirical quantiles
        self._X = X.copy()
        self._y = y.copy()
        self._mean = y.mean().values
        self._std = y.std(ddof=1).values
        self._n = len(y)
        self._y_cols = y.columns if hasattr(y, "columns") else ["y"]
        return self

    def _predict(self, X):
        # Predict mean for all X
        mean = pd.DataFrame(
            np.tile(self._mean, (len(X), 1)), index=X.index, columns=self._y_cols
        )
        return mean

    def _predict_interval(self, X, coverage):
        # Interval shrinks as n increases
        n = self._n
        mean = np.tile(self._mean, (len(X), 1))
        std = np.tile(self._std, (len(X), 1))
        intervals = []
        for c in coverage:
            z = abs(np.percentile(np.random.normal(size=100000), 100 * (0.5 + c / 2)))
            half_width = z * std / np.sqrt(n)
            lower = mean - half_width
            upper = mean + half_width
            intervals.append((lower, upper))
        # Build MultiIndex columns
        arrays = [
            np.repeat(self._y_cols, len(coverage) * 2),
            np.tile(np.repeat(coverage, 2), len(self._y_cols)),
            np.tile(["lower", "upper"], len(self._y_cols) * len(coverage)),
        ]
        columns = pd.MultiIndex.from_arrays(arrays, names=["var", "coverage", "bound"])
        data = np.hstack(
            [np.column_stack([lower, upper]) for lower, upper in intervals]
        )
        return pd.DataFrame(data, index=X.index, columns=columns)

    def _predict_quantiles(self, X, alpha):
        # Use empirical quantiles from training data
        quantiles = []
        for a in alpha:
            q = np.percentile(self._y.values, 100 * a, axis=0)
            quantiles.append(np.tile(q, (len(X), 1)))
        # Build MultiIndex columns
        arrays = [
            np.repeat(self._y_cols, len(alpha)),
            np.tile(alpha, len(self._y_cols)),
        ]
        columns = pd.MultiIndex.from_arrays(arrays, names=["var", "alpha"])
        data = np.hstack(quantiles)
        return pd.DataFrame(data, index=X.index, columns=columns)

        @classmethod
        def get_test_params(cls, parameter_set="default"):
            """Return testing parameter sets for automated tests.

            Returns two parameter sets: one for mean/sd, one for quantile method.
            """
            return [
                {"method": "mean_sd"},
                {"method": "quantile"},
            ]