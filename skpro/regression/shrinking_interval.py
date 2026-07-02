"""Shrinking normal interval regressor with a static quantile baseline."""

from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from skpro.regression.base import BaseProbaRegressor


class ShrinkingNormalIntervalRegressor(BaseProbaRegressor):
    """Probabilistic regressor with shrinking intervals in ``mean_sd`` mode.

    Parameters
    ----------
    method : str, default="mean_sd"
        "mean_sd" for normal-approximation intervals that shrink with ``n``;
        "quantile" for a static empirical-quantile baseline.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from skpro.regression.shrinking_interval import ShrinkingNormalIntervalRegressor
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> y = pd.DataFrame(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> reg = ShrinkingNormalIntervalRegressor(method="mean_sd")
    >>> reg.fit(X_train, y_train)
    ShrinkingNormalIntervalRegressor(...)
    >>> y_pred = reg.predict(X_test)
    >>> intervals = reg.predict_interval(X_test, coverage=[0.9])
    >>> quantiles = reg.predict_quantiles(X_test, alpha=[0.05, 0.5, 0.95])
    """

    _tags = {
        "authors": ["arnavk23"],
        "capability:multioutput": False,
        "capability:missing": True,
    }

    def __init__(self, method: str = "mean_sd"):
        if method not in ("mean_sd", "quantile"):
            raise ValueError(f"method must be 'mean_sd' or 'quantile', got {method}")
        self.method = method
        super().__init__()

    def _fit(self, X: pd.DataFrame, y: pd.DataFrame, C: Optional[Any] = None):
        # Input validation
        if not isinstance(y, (pd.DataFrame, pd.Series, np.ndarray)) or len(y) == 0:
            raise ValueError("y must be a non-empty DataFrame, Series, or ndarray")
        self._X = X.copy()
        self._y = y.copy() if hasattr(y, "copy") else np.array(y)
        self._mean = np.asarray(
            y.mean().values if hasattr(y, "mean") else np.mean(y, axis=0)
        )
        self._std = np.asarray(
            y.std(ddof=1).values if hasattr(y, "std") else np.std(y, ddof=1, axis=0)
        )
        if np.any(~np.isfinite(self._std)):
            self._std = np.nan_to_num(self._std, nan=0.0, posinf=0.0, neginf=0.0)
        self._n = len(y)
        self._y_cols = y.columns if hasattr(y, "columns") else ["y"]
        return self

    def _predict(self, X: pd.DataFrame) -> pd.DataFrame:
        # Predict mean for all X (featureless baseline)
        mean = pd.DataFrame(
            np.tile(self._mean, (len(X), 1)), index=X.index, columns=self._y_cols
        )
        return mean

    def _predict_interval(self, X: pd.DataFrame, coverage: List[float]) -> pd.DataFrame:
        n = self._n
        mean = np.tile(self._mean, (len(X), 1))
        std = np.tile(self._std, (len(X), 1))
        intervals = []
        for c in coverage:
            if not (0 < c <= 1):
                raise ValueError(f"coverage must be in (0, 1], got {c}")
            if self.method == "mean_sd":
                z = abs(norm.ppf((1 + c) / 2))
                half_width = z * std / np.sqrt(n)
                lower = mean - half_width
                upper = mean + half_width
            elif self.method == "quantile":
                # Use empirical quantiles from training data, not shrinking
                lower = np.tile(
                    np.percentile(self._y.values, 100 * (1 - c) / 2, axis=0),
                    (len(X), 1),
                )
                upper = np.tile(
                    np.percentile(self._y.values, 100 * (1 + c) / 2, axis=0),
                    (len(X), 1),
                )
            else:
                raise NotImplementedError(f"Unknown method: {self.method}")
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

    def _predict_quantiles(self, X: pd.DataFrame, alpha: List[float]) -> pd.DataFrame:
        quantiles = []
        for a in alpha:
            if not (0 <= a <= 1):
                raise ValueError(f"alpha must be in [0, 1], got {a}")
            if self.method == "quantile":
                q = np.percentile(self._y.values, 100 * a, axis=0)
            elif self.method == "mean_sd":
                # For mean_sd, return mean for median
                # or mean +/- z*std/sqrt(n) for tails
                if a == 0.5:
                    q = self._mean
                else:
                    z = abs(norm.ppf(a))
                    q = self._mean + np.sign(a - 0.5) * z * self._std / np.sqrt(self._n)
            else:
                raise NotImplementedError(f"Unknown method: {self.method}")
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
    def get_test_params(cls, parameter_set: str = "default"):
        """Return testing parameter sets for automated tests.

        Returns two parameter sets: one for mean/sd, one for quantile method.
        """
        return [
            {"method": "mean_sd"},
            {"method": "quantile"},
        ]
