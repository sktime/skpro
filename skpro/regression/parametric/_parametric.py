"""Parametric distribution wrapper for sklearn regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["arnavk23"]
__all__ = ["ParametricRegressor"]

import numpy as np
import pandas as pd
from sklearn import clone

from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class ParametricRegressor(BaseProbaRegressor):
    """Parametric probabilistic regressor wrapping a point predictor.

    Wraps an ``sklearn`` regressor and turns it into a parametric probabilistic
    regressor by fitting a specified parametric distribution family.

    The wrapped regressor predicts the location parameter (e.g., mean) of the
    distribution. The scale parameter is either fixed or estimated from the
    residuals on the training set.

    Parameters
    ----------
    estimator : sklearn regressor
        Regressor predicting the location parameter (e.g., mean).
    distr : str or BaseDistribution class, default="Normal"
        Distribution family to use for probabilistic predictions.
        String options include: "Normal", "Laplace", "Cauchy", "t", "Poisson",
        "Gamma", "Exponential", "LogNormal".
    scale : float or None, default=None
        Fixed value for the scale parameter. If None, the scale is estimated
        from the training residuals (for distributions with scale parameter).
    distr_params : dict, default={}
        Additional fixed parameters to pass to the distribution.
        Keys must be valid parameter names for the chosen distribution.

    Attributes
    ----------
    estimator_ : sklearn regressor
        Fitted clone of the wrapped estimator.
    scale_ : float
        Estimated or fixed scale parameter (for distributions with scale).

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from skpro.regression.parametric import ParametricRegressor
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg = ParametricRegressor(LinearRegression(), distr="Normal")
    >>> reg.fit(X_train, y_train)
    ParametricRegressor(...)
    >>> y_pred_proba = reg.predict_proba(X_test)
    """

    _tags = {
        "authors": ["arnavk23"],
        "capability:missing": True,
    }

    # Distribution name mappings
    _DISTR_MAP = {
        "normal": ("Normal", "mu", "sigma"),
        "laplace": ("Laplace", "mu", "scale"),
        "cauchy": ("Cauchy", "loc", "scale"),
        "t": ("TDistribution", "mu", "sigma"),
        "poisson": ("Poisson", "mu", None),
        "gamma": ("Gamma", "alpha", "beta"),
        "exponential": ("Exponential", "rate", None),
        "lognormal": ("LogNormal", "mu", "sigma"),
    }

    def __init__(self, estimator, distr="Normal", scale=None, distr_params=None):
        self.estimator = estimator
        self.distr = distr
        self.scale = scale
        self.distr_params = distr_params if distr_params is not None else {}
        super().__init__()

    def _get_distribution_class(self):
        """Get the distribution class based on distr parameter."""
        if isinstance(self.distr, str):
            distr_lower = self.distr.lower()
            if distr_lower not in self._DISTR_MAP:
                raise ValueError(
                    f"Unknown distribution: {self.distr}. "
                    f"Must be one of {list(self._DISTR_MAP.keys())}"
                )
            distr_name, loc_name, scale_name = self._DISTR_MAP[distr_lower]

            # Import the distribution class
            if distr_name == "Normal":
                from skpro.distributions.normal import Normal

                distr_class = Normal
            elif distr_name == "Laplace":
                from skpro.distributions.laplace import Laplace

                distr_class = Laplace
            elif distr_name == "Cauchy":
                # For Cauchy, we'll use Normal as approximation for now
                from skpro.distributions.normal import Normal

                distr_class = Normal
                scale_name = "sigma"
                loc_name = "mu"
            elif distr_name == "TDistribution":
                from skpro.distributions.t import TDistribution

                distr_class = TDistribution
            elif distr_name == "Poisson":
                from skpro.distributions.poisson import Poisson

                distr_class = Poisson
            elif distr_name == "Gamma":
                from skpro.distributions.gamma import Gamma

                distr_class = Gamma
            elif distr_name == "Exponential":
                from skpro.distributions.exponential import Exponential

                distr_class = Exponential
            elif distr_name == "LogNormal":
                from skpro.distributions.lognormal import LogNormal

                distr_class = LogNormal
            else:
                from skpro.distributions.normal import Normal

                distr_class = Normal

            return distr_class, loc_name, scale_name
        else:
            # Assume it's a BaseDistribution class
            return self.distr, None, None

    def _fit(self, X, y):
        """Fit the parametric regressor.

        Parameters
        ----------
        X : pandas DataFrame
            Feature instances to fit regressor to.
        y : pandas DataFrame
            Labels to fit regressor to (must be same length as X).

        Returns
        -------
        self : reference to self
        """
        # Coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        # Store column names
        self._y_cols = y.columns

        # Fit the base estimator
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.ravel(y))

        # Get distribution metadata
        (
            self._distr_class,
            self._loc_name,
            self._scale_name,
        ) = self._get_distribution_class()

        # Estimate scale from residuals if not fixed
        if self.scale is None and self._scale_name is not None:
            y_pred = self.estimator_.predict(X)
            residuals = np.ravel(y) - y_pred

            # Estimate scale parameter (standard deviation for Normal, etc.)
            if self.distr.lower() in ["normal", "lognormal", "t"]:
                self.scale_ = np.std(residuals)
            elif self.distr.lower() in ["laplace", "cauchy"]:
                # For Laplace, scale = mean absolute deviation
                self.scale_ = np.mean(np.abs(residuals - np.mean(residuals)))
            else:
                self.scale_ = np.std(residuals)

            # Ensure scale is positive
            self.scale_ = max(self.scale_, 1e-10)
        else:
            self.scale_ = self.scale if self.scale is not None else 1.0

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame
            Data to predict labels for.

        Returns
        -------
        y : pandas DataFrame
            Predictions of target values for X.
        """
        X = prep_skl_df(X, copy_df=True)
        y_pred = self.estimator_.predict(X)

        return pd.DataFrame(y_pred, index=X.index, columns=self._y_cols)

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame
            Data to predict labels for.

        Returns
        -------
        y_proba : skpro BaseDistribution
            Predictive distribution for X.
        """
        X = prep_skl_df(X, copy_df=True)

        # Get point predictions (location parameter)
        y_pred = self.estimator_.predict(X).reshape(-1, 1)

        # Build distribution parameters
        params = dict(self.distr_params)
        params["index"] = X.index
        params["columns"] = self._y_cols

        # Set location parameter
        if self._loc_name:
            params[self._loc_name] = y_pred

        # Set scale parameter if applicable
        if self._scale_name:
            scale_broadcast = np.full_like(y_pred, self.scale_)
            params[self._scale_name] = scale_broadcast

        # Create and return distribution
        return self._distr_class(**params)

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

        params1 = {"estimator": LinearRegression(), "distr": "Normal"}
        params2 = {
            "estimator": LinearRegression(),
            "distr": "Laplace",
            "scale": 1.0,
        }

        return [params1, params2]
