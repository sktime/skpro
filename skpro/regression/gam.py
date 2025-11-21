# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Generalized Additive Models (GAM) Regressor."""

__author__ = ["Omswastik-11"]

import numpy as np
import pandas as pd

# from skpro.distributions.inversegaussian import InverseGaussian
from skpro.distributions.binomial import Binomial
from skpro.distributions.gamma import Gamma
from skpro.distributions.normal import Normal
from skpro.distributions.poisson import Poisson
from skpro.regression.base import BaseProbaRegressor


class GAMRegressor(BaseProbaRegressor):
    """Generalized Additive Model (GAM) Regressor.

    Wraps the `pygam` library.

    Parameters
    ----------
    terms : expression specifying terms to model, optional (default='auto')
        By default a univariate spline term will be allocated for each feature.
    distribution : str or pygam.Distribution, optional (default='normal')
        Distribution to use in the model.
        Supported strings: 'normal', 'binomial', 'poisson', 'gamma'.
    link : str or pygam.Link, optional (default='identity')
        Link function to use in the model.
    max_iter : int, optional (default=100)
        Maximum number of iterations allowed for the solver to converge.
    tol : float, optional (default=1e-4)
        Tolerance for stopping criteria.
    callbacks : list of str or list of CallBack objects, optional
        Names of callback objects to call during the optimization loop.
        Default is ['deviance', 'diffs'].
    fit_intercept : bool, optional (default=True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    verbose : bool, optional (default=False)
        whether to show pyGAM warnings.

    Attributes
    ----------
    estimator_ : pygam.GAM
        The fitted pygam estimator.
    """

    _tags = {
        "authors": ["Omswastik-11"],
        "maintainers": ["fkiraly"],
        "python_dependencies": ["pygam"],
        "capability:multioutput": False,
        "capability:missing": True,
        "capability:update": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(
        self,
        terms="auto",
        distribution="normal",
        link="identity",
        max_iter=100,
        tol=1e-4,
        callbacks=None,
        fit_intercept=True,
        verbose=False,
    ):
        self.terms = terms
        self.distribution = distribution
        self.link = link
        self.max_iter = max_iter
        self.tol = tol
        self.callbacks = callbacks
        self.fit_intercept = fit_intercept
        self.verbose = verbose

        super().__init__()

    def _fit(self, X, y):
        """Fit regressor to training data.

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        from pygam import GAM

        # pygam expects numpy arrays
        X_np = X.values
        y_np = y.values.flatten()

        self._y_cols = y.columns

        # Handle callbacks default
        callbacks = self.callbacks
        if callbacks is None:
            callbacks = ["deviance", "diffs"]

        self.estimator_ = GAM(
            terms=self.terms,
            distribution=self.distribution,
            link=self.link,
            max_iter=self.max_iter,
            tol=self.tol,
            callbacks=callbacks,
            fit_intercept=self.fit_intercept,
            verbose=self.verbose,
        )

        self.estimator_.fit(X_np, y_np)

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame
            labels predicted for X
        """
        X_np = X.values
        y_pred_np = self.estimator_.predict(X_np)

        return pd.DataFrame(y_pred_np, index=X.index, columns=self._y_cols)

    def _get_distribution_name(self, dist):
        """Extract distribution name from pyGAM estimator."""
        if dist is None:
            return "normal"
        if isinstance(dist, str):
            return dist.lower()
        if hasattr(dist, "name"):
            return dist.name
        if hasattr(dist, "__name__"):
            return dist.__name__
        if hasattr(dist, "__class__"):
            return dist.__class__.__name__.lower()
        return str(dist).lower()

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame
            data to predict labels for

        Returns
        -------
        y_pred : skpro BaseDistribution
            labels predicted for X
        """
        X_np = X.values
        mu = self.estimator_.predict_mu(X_np)

        # Ensure mu is 2D if it's 1D, to match (n_samples, n_outputs)
        if mu.ndim == 1:
            mu = mu.reshape(-1, 1)

        dist = getattr(self.estimator_, "distribution", None)
        if dist is None:
            dist = getattr(self.estimator_, "_distribution", None)

        dist_name = self._get_distribution_name(dist)

        # Map common names
        dist_map = {
            "normal": "normal",
            "gaussian": "normal",
            "poisson": "poisson",
            "gamma": "gamma",
            "binomial": "binomial",
        }
        dist_name = dist_map.get(dist_name, "normal")

        # Get scale from statistics if available, else from estimator.distribution
        if (
            hasattr(self.estimator_, "statistics_")
            and "scale" in self.estimator_.statistics_
        ):
            scale = self.estimator_.statistics_["scale"]
        elif hasattr(self.estimator_, "distribution") and hasattr(
            self.estimator_.distribution, "scale"
        ):
            scale = self.estimator_.distribution.scale
        else:
            scale = 1.0  # Default fallback

        index = X.index
        columns = self._y_cols

        if dist_name == "normal":
            # Normal distribution
            # pygam scale is variance (sigma^2) (dispersion)
            sigma = np.sqrt(scale)
            return Normal(mu=mu, sigma=sigma, index=index, columns=columns)

        elif dist_name == "poisson":
            # Poisson distribution
            # Parameter is mu. Scale is 1 (or ignored if overdispersion).
            return Poisson(mu=mu, index=index, columns=columns)

        elif dist_name == "binomial":
            # Binomial distribution
            # Parameters: n (levels), p.
            # mu = n * p => p = mu / n.
            levels = 1
            if hasattr(self.estimator_.distribution, "levels"):
                levels = self.estimator_.distribution.levels

            p = mu / levels
            # Clip p to [0, 1] to avoid numerical issues
            p = np.clip(p, 0, 1)
            return Binomial(n=levels, p=p, index=index, columns=columns)

        elif dist_name == "gamma":
            # Gamma distribution
            # pygam scale is dispersion phi = 1/alpha
            # alpha = 1/phi
            # beta = 1/(phi * mu)
            alpha = 1.0 / scale
            beta = 1.0 / (scale * mu)
            return Gamma(alpha=alpha, beta=beta, index=index, columns=columns)

        else:
            # Fallback to Normal
            sigma = np.sqrt(scale)
            return Normal(mu=mu, sigma=sigma, index=index, columns=columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params = [
            {"distribution": "normal", "terms": "auto"},
            {"distribution": "poisson", "terms": "auto"},
            {"distribution": "gamma", "terms": "auto"},
        ]
        return params
