"""MAPIE Conformalized Quantile Regressor."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Omswastik-11", "vtaquet", "vincentblot28", "TMorzadec", "gmartinonQM"]

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class MapieConformalizedQuantileRegressor(BaseProbaRegressor):
    """MAPIE Conformalized Quantile Regressor.

    Direct interface to ``mapie.regression.ConformalizedQuantileRegressor`` from the
    ``mapie`` package (>= 1.0).

    Note: Unlike other conformal regressors, CQR requires the confidence level
    to be specified at training time because the quantile regressors are trained
    for specific quantiles. The `predict_interval` method will only return
    valid intervals for the specified `confidence_level`.

    Parameters
    ----------
    estimator : sklearn regressor, default=None
        Regressor with scikit-learn compatible API.
        Must support quantile loss.
    confidence_level : float, default=0.9
        Target confidence level for prediction intervals.
        The quantile regressors are trained for quantiles (1-confidence_level)/2
        and (1+confidence_level)/2.
    test_size : float or int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the calibration set.
        If int, represents the absolute number of calibration samples.
    random_state : int, RandomState instance or None, default=None
    """

    _tags = {
        "authors": [
            "Omswastik-11",
            "vtaquet",
            "vincentblot28",
            "TMorzadec",
            "gmartinonQM",
        ],
        "maintainers": ["fkiraly", "Omswastik-11"],
        "python_dependencies": ["MAPIE>=1.0"],
        "capability:missing": True,
    }

    def __init__(
        self,
        estimator=None,
        confidence_level=0.9,
        test_size=0.2,
        random_state=None,
    ):
        self.estimator = estimator
        self.confidence_level = confidence_level
        self.test_size = test_size
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        from mapie.regression import ConformalizedQuantileRegressor

        X = prep_skl_df(X, copy_df=True)

        self.mapie_est_ = ConformalizedQuantileRegressor(
            estimator=self.estimator,
            confidence_level=self.confidence_level,
        )

        # Convert y to 1D array for sklearn compatibility
        y_values = y.values.ravel() if hasattr(y, "values") else np.asarray(y).ravel()

        # Split data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y_values, test_size=self.test_size, random_state=self.random_state
        )

        self.mapie_est_.fit(X_train, y_train)
        self.mapie_est_.conformalize(X_cal, y_cal)

        self._y_cols = y.columns if hasattr(y, "columns") else ["y"]

        return self

    def _predict(self, X):
        X = prep_skl_df(X, copy_df=True)
        y_pred_np = self.mapie_est_.predict(X)
        return pd.DataFrame(y_pred_np, index=X.index, columns=self._y_cols)

    def _predict_interval(self, X, coverage):
        X = prep_skl_df(X, copy_df=True)
        return _predict_interval_cqr(self, X, coverage)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        params1 = {
            "estimator": GradientBoostingRegressor(loss="quantile", alpha=0.5),
            "confidence_level": 0.5,
        }
        params2 = {
            "estimator": GradientBoostingRegressor(
                loss="quantile", alpha=0.5, n_estimators=50, random_state=0
            ),
            "confidence_level": 0.5,
            "test_size": 0.5,
            "random_state": 42,
        }
        params3 = {
            "estimator": GradientBoostingRegressor(
                loss="quantile", alpha=0.5, max_depth=3, random_state=0
            ),
            "confidence_level": 0.5,
            "test_size": 0.4,
            "random_state": 0,
        }
        return [params1, params2, params3]


def _predict_interval_cqr(self, X, coverage):
    """Predict intervals for ConformalizedQuantileRegressor.

    CQR is trained for a specific confidence level, so we use the trained
    intervals and return them for any requested coverage level.
    Note: The returned intervals are only truly valid for the confidence_level
    the model was trained with.
    """
    results = {}

    _, intervals = self.mapie_est_.predict_interval(X)

    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]

    for cov in coverage:
        results[(cov, "lower")] = lower
        results[(cov, "upper")] = upper

    return _build_interval_df(self, X, coverage, results)


def _build_interval_df(self, X, coverage, results):
    """Build the interval DataFrame from results."""
    index = X.index
    columns = pd.MultiIndex.from_product(
        [self._y_cols, coverage, ["lower", "upper"]],
    )

    values = np.zeros((len(X), len(coverage) * 2))
    for i, cov in enumerate(coverage):
        values[:, i * 2] = results[(cov, "lower")]
        values[:, i * 2 + 1] = results[(cov, "upper")]

    return pd.DataFrame(values, index=index, columns=columns)
