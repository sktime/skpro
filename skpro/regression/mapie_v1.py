"""Interface adapters for MAPIE regressor >= 1.0."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Omswastik-11"]  # interface only. MAPIE authors in mapie package
__all__ = [
    "MapieSplitConformalRegressor",
    "MapieCrossConformalRegressor",
    "MapieJackknifeAfterBootstrapRegressor",
    "MapieConformalizedQuantileRegressor",
]

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skpro.regression.base import BaseProbaRegressor


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
        "authors": ["Omswastik-11"],
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

        # Split data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.mapie_est_.fit(X_train, y_train)
        self.mapie_est_.conformalize(X_cal, y_cal)

        # Store y columns for predict
        self._y_cols = y.columns if hasattr(y, "columns") else ["y"]

        return self

    def _predict(self, X):
        y_pred_np = self.mapie_est_.predict(X)
        return pd.DataFrame(y_pred_np, index=X.index, columns=self._y_cols)

    def _predict_interval(self, X, coverage):
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
        from sklearn.linear_model import LinearRegression

        params1 = {}
        params2 = {
            "estimator": LinearRegression(),
            "test_size": 0.3,
            "method": "absolute",
            "random_state": 42,
        }
        return [params1, params2]


class MapieCrossConformalRegressor(BaseProbaRegressor):
    """MAPIE Cross Conformal Regressor.

    Direct interface to ``mapie.regression.CrossConformalRegressor`` from the
    ``mapie`` package (>= 1.0).

    Parameters
    ----------
    estimator : sklearn regressor, default=None
        Regressor with scikit-learn compatible API.
    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
    method : str, default="plus"
        Method to choose for prediction interval estimates.
        "plus": Jackknife+
        "minmax": Jackknife-minmax
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        The verbosity level.
    random_state : int, RandomState instance or None, default=None
    """

    _tags = {
        "authors": ["Omswastik-11"],
        "maintainers": ["fkiraly", "Omswastik-11"],
        "python_dependencies": ["MAPIE>=1.0"],
        "capability:missing": True,
    }

    def __init__(
        self,
        estimator=None,
        cv=None,
        method="plus",
        n_jobs=None,
        verbose=0,
        random_state=None,
    ):
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        from mapie.regression import CrossConformalRegressor

        self.mapie_est_ = CrossConformalRegressor(
            estimator=self.estimator,
            cv=self.cv,
            method=self.method,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            confidence_level=0.5,  # Dummy
        )

        self.mapie_est_.fit_conformalize(X, y)
        self._y_cols = y.columns if hasattr(y, "columns") else ["y"]

        return self

    def _predict(self, X):
        y_pred_np = self.mapie_est_.predict(X)
        return pd.DataFrame(y_pred_np, index=X.index, columns=self._y_cols)

    def _predict_interval(self, X, coverage):
        return _predict_interval_mapie(self, X, coverage)

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
        from sklearn.linear_model import LinearRegression

        params1 = {}
        params2 = {
            "estimator": LinearRegression(),
            "cv": 3,
            "method": "plus",
            "random_state": 42,
        }
        return [params1, params2]


class MapieJackknifeAfterBootstrapRegressor(BaseProbaRegressor):
    """MAPIE Jackknife+ after Bootstrap Regressor.

    Direct interface to ``mapie.regression.JackknifeAfterBootstrapRegressor`` from the
    ``mapie`` package (>= 1.0).

    Parameters
    ----------
    estimator : sklearn regressor, default=None
        Regressor with scikit-learn compatible API.
    cv : int or Subsample, default=30
        Number of bootstrap resamples or a Subsample instance.
    method : str, default="plus"
        Method to choose for prediction interval estimates.
        "plus": Jackknife+
        "minmax": Jackknife-minmax
    agg_function : str, default="mean"
        Aggregation function for predictions ("mean" or "median").
    n_jobs : int, default=None
        Number of jobs to run in parallel.
    verbose : int, default=0
        The verbosity level.
    random_state : int, RandomState instance or None, default=None
    """

    _tags = {
        "authors": ["Omswastik-11"],
        "maintainers": ["fkiraly", "Omswastik-11"],
        "python_dependencies": ["MAPIE>=1.0"],
        "capability:missing": True,
    }

    def __init__(
        self,
        estimator=None,
        cv=30,
        method="plus",
        agg_function="mean",
        n_jobs=None,
        verbose=0,
        random_state=None,
    ):
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self.agg_function = agg_function
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state
        super().__init__()

    def _fit(self, X, y):
        from mapie.regression import JackknifeAfterBootstrapRegressor

        self.mapie_est_ = JackknifeAfterBootstrapRegressor(
            estimator=self.estimator,
            resampling=self.cv,
            method=self.method,
            aggregation_method=self.agg_function,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            confidence_level=0.5,  # Dummy
        )

        self.mapie_est_.fit_conformalize(X, y)
        self._y_cols = y.columns if hasattr(y, "columns") else ["y"]

        return self

    def _predict(self, X):
        y_pred_np = self.mapie_est_.predict(X)
        return pd.DataFrame(y_pred_np, index=X.index, columns=self._y_cols)

    def _predict_interval(self, X, coverage):
        return _predict_interval_mapie(self, X, coverage)

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
        from sklearn.linear_model import LinearRegression

        params1 = {}
        params2 = {
            "estimator": LinearRegression(),
            "cv": 10,
            "method": "plus",
            "agg_function": "median",
            "random_state": 42,
        }
        return [params1, params2]


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
        "authors": ["Omswastik-11"],
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

        self.mapie_est_ = ConformalizedQuantileRegressor(
            estimator=self.estimator,
            confidence_level=self.confidence_level,
        )

        # Split data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        self.mapie_est_.fit(X_train, y_train)
        self.mapie_est_.conformalize(X_cal, y_cal)

        self._y_cols = y.columns if hasattr(y, "columns") else ["y"]

        return self

    def _predict(self, X):
        y_pred_np = self.mapie_est_.predict(X)
        return pd.DataFrame(y_pred_np, index=X.index, columns=self._y_cols)

    def _predict_interval(self, X, coverage):
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

        # Use lower confidence levels for testing with small datasets
        # CQR requires 1/alpha calibration samples, so confidence_level=0.5
        # requires only 2 samples minimum
        params1 = {
            "estimator": GradientBoostingRegressor(loss="quantile", alpha=0.5),
            "confidence_level": 0.5,
        }
        params2 = {
            "estimator": GradientBoostingRegressor(loss="quantile", alpha=0.5),
            "confidence_level": 0.7,
            "test_size": 0.4,
            "random_state": 42,
        }
        return [params1, params2]


def _predict_interval_split_conformal(self, X, coverage):
    """Predict intervals for SplitConformalRegressor."""
    results = {}
    # SplitConformalRegressor stores alphas in _alphas list
    original_alphas = self.mapie_est_._alphas.copy()

    try:
        for cov in coverage:
            alpha = 1 - cov
            self.mapie_est_._alphas = [alpha]

            _, intervals = self.mapie_est_.predict_interval(X)

            # Flatten the output - it may have extra dimensions
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


def _predict_interval_mapie(self, X, coverage):
    """Predict intervals with dynamic coverage."""
    results = {}
    # Store original alphas to restore later
    original_alphas = self.mapie_est_._alphas.copy()

    try:
        for cov in coverage:
            # mapie stores alphas = 1 - confidence_level
            # skpro coverage IS confidence_level (1-alpha)
            alpha = 1 - cov

            # Set the internal _alphas attribute
            self.mapie_est_._alphas = [alpha]

            # predict_interval returns (y_pred, intervals)
            # intervals shape: (n_samples, 2, n_alpha) or (n_samples, 2)
            _, intervals = self.mapie_est_.predict_interval(X)

            # Flatten the output - it may have extra dimensions
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


def _predict_interval_cqr(self, X, coverage):
    """Predict intervals for ConformalizedQuantileRegressor.

    CQR is trained for a specific confidence level, so we use the trained
    intervals and return them for any requested coverage level.
    Note: The returned intervals are only truly valid for the confidence_level
    the model was trained with.
    """
    results = {}

    # CQR predict_interval doesn't take alpha - returns intervals for trained conf
    _, intervals = self.mapie_est_.predict_interval(X)

    # intervals shape: (n_samples, 2, 1)
    lower = intervals[:, 0, 0]
    upper = intervals[:, 1, 0]

    # Return the same intervals for all requested coverages
    # Note: only truly valid for self.confidence_level
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
