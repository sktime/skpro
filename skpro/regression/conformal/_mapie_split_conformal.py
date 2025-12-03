"""MAPIE Split Conformal Regressor."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Omswastik-11", "vtaquet", "vincentblot28", "TMorzadec", "gmartinonQM"]

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class MapieSplitConformalRegressor(BaseProbaRegressor):
    """MAPIE Split Conformal Regressor.

    Direct interface to ``mapie.regression.SplitConformalRegressor`` from the
    ``mapie`` package (>= 1.0).

    Parameters
    ----------
    estimator : sklearn regressor, default=None
        Regressor with scikit-learn compatible API.
        If None, defaults to LinearRegression.
    test_size : float or int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the calibration set.
        If int, represents the absolute number of calibration samples.
    method : str, default="absolute"
        Conformity score method.
        "absolute": AbsoluteConformityScore
        "gamma": GammaConformityScore
    random_state : int, RandomState instance or None, default=None
        Controls the shuffling applied to the data before applying the split.
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
        test_size=0.2,
        method="absolute",
        random_state=None,
    ):
        self.estimator = estimator
        self.test_size = test_size
        self.method = method
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        from mapie.conformity_scores import (
            AbsoluteConformityScore,
            GammaConformityScore,
        )
        from mapie.regression import SplitConformalRegressor

        X = prep_skl_df(X, copy_df=True)

        # Handle conformity score
        if self.method == "absolute":
            conf_score = AbsoluteConformityScore()
        elif self.method == "gamma":
            conf_score = GammaConformityScore()
        else:
            # Allow passing custom score object if needed, or just support strings
            conf_score = self.method

        self.mapie_est_ = SplitConformalRegressor(
            estimator=self.estimator,
            conformity_score=conf_score,
            confidence_level=0.5,  # Dummy value
            prefit=False,
        )

        # Convert y to 1D array for sklearn compatibility
        y_values = y.values.ravel() if hasattr(y, "values") else np.asarray(y).ravel()

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
        return _predict_interval_split_conformal(self, X, coverage)

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
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression, Ridge

        params1 = {
            "estimator": LinearRegression(),
            "test_size": 0.7,
        }
        params2 = {
            "estimator": Ridge(alpha=1.0),
            "test_size": 0.7,
            "method": "absolute",
            "random_state": 42,
        }
        params3 = {
            "estimator": RandomForestRegressor(n_estimators=5, random_state=0),
            "test_size": 0.7,
            "method": "absolute",
            "random_state": 0,
        }
        return [params1, params2, params3]


def _predict_interval_split_conformal(self, X, coverage):
    """Predict intervals for SplitConformalRegressor."""
    results = {}
    original_alphas = self.mapie_est_._alphas.copy()

    try:
        for cov in coverage:
            alpha = 1 - cov
            self.mapie_est_._alphas = [alpha]

            _, intervals = self.mapie_est_.predict_interval(X)

            intervals = np.squeeze(intervals)
            if intervals.ndim == 2:
                lower = intervals[:, 0]
                upper = intervals[:, 1]
            elif intervals.ndim == 3:
                lower = intervals[:, 0, 0]
                upper = intervals[:, 1, 0]
            else:
                lower = intervals[:, 0]
                upper = intervals[:, 1]

            results[(cov, "lower")] = lower
            results[(cov, "upper")] = upper

    finally:
        self.mapie_est_._alphas = original_alphas

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
