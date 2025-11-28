# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Generalized Additive Models (GAM) Regressor using pygam."""

__author__ = ["Omswastik-11", "ravjot07"]

import numpy as np
import pandas as pd

# from skpro.distributions.inversegaussian import InverseGaussian
from skpro.distributions.binomial import Binomial
from skpro.distributions.gamma import Gamma
from skpro.distributions.normal import Normal
from skpro.distributions.poisson import Poisson
from skpro.regression.base import BaseProbaRegressor


class GAMRegressor(BaseProbaRegressor):
    """Generalized Additive Model (GAM) Regressor using pygam.

    Wraps the ``pygam`` library to provide probabilistic predictions using
    Generalized Additive Models with various distribution families.

    The ``distribution``
    parameter determines which skpro distribution will be returned in
    ``predict_proba``.

    Parameters
    ----------
    terms : expression specifying terms to model, optional (default='auto')
        By default a univariate spline term will be allocated for each feature.
        Can be a ``pygam`` terms expression for custom model specification.

    distribution : str or pygam.Distribution, optional (default='Normal')
        Distribution family to use in the model.
        Supported strings (case-insensitive):

        * ``'Normal'`` or ``'Gaussian'`` - Normal/Gaussian distribution
        * ``'Poisson'`` - Poisson distribution for count data
        * ``'Gamma'`` - Gamma distribution for positive continuous data
        * ``'Binomial'`` - Binomial distribution for binary/proportion data

        Alternatively, can pass a ``pygam.Distribution`` object directly.

    link : str or pygam.Link, optional (default='identity')
        Link function to use in the model. Common options:

        * ``'identity'`` - for Normal distribution
        * ``'log'`` - for Poisson, Gamma distributions
        * ``'logit'`` - for Binomial distribution
        * ``'inverse'`` - for Gamma distribution

        Alternatively, can pass a ``pygam.Link`` object directly.

    max_iter : int, optional (default=100)
        Maximum number of iterations allowed for the solver to converge.
    tol : float, optional (default=1e-4)
        Tolerance for stopping criteria.
    callbacks : list of str or list of CallBack objects, optional
        Names of callback objects to call during the optimization loop.
        Default is ``['deviance', 'diffs']``.
    fit_intercept : bool, optional (default=True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    verbose : bool, optional (default=False)
        Whether to show pyGAM warnings.

    Attributes
    ----------
    estimator_ : pygam.GAM
        The fitted pygam estimator.

    Examples
    --------
    >>> from skpro.regression.gam import GAMRegressor
    >>> from sklearn.datasets import make_regression
    >>> import pandas as pd
    >>>
    >>> X, y = make_regression(n_samples=100, n_features=3, random_state=42)
    >>> X = pd.DataFrame(X, columns=['f1', 'f2', 'f3'])
    >>> y = pd.DataFrame(y, columns=['target'])
    >>> y_positive = y.abs() + 1  # ensure positive targets for Poisson/Gamma
    >>>
    >>> # Normal distribution (default)
    >>> gam_normal = GAMRegressor(distribution='Normal')
    >>> gam_normal.fit(X, y)
    GAMRegressor(...)
    >>>
    >>> # Poisson distribution for count data
    >>> gam_poisson = GAMRegressor(distribution='Poisson', link='log')
    >>> gam_poisson.fit(X, y_positive)
    GAMRegressor(...)
    >>>
    >>> # Gamma distribution for positive continuous data
    >>> gam_gamma = GAMRegressor(distribution='Gamma', link='log')
    >>> gam_gamma.fit(X, y_positive)
    GAMRegressor(...)
    """

    _tags = {
        "authors": ["dswah", "Omswastik-11", "ravjot07"],
        # dswah for pygam package
        "maintainers": ["fkiraly", "Omswastik-11", "dswah"],
        "python_dependencies": ["pygam"],
        "capability:multioutput": False,
        "capability:missing": True,
        "capability:update": False,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
        "tests:vm": True,
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

        dist_name = self._get_distribution_name(self.distribution)

        # Map common names to skpro distribution names
        dist_map = {
            "normal": "normal",
            "gaussian": "normal",
            "poisson": "poisson",
            "gamma": "gamma",
            "binomial": "binomial",
            "normaldist": "normal",
            "poissondist": "poisson",
            "gammadist": "gamma",
            "binomialdist": "binomial",
        }
        dist_name = dist_map.get(dist_name, "normal")

        self._dist_name = dist_name

        self.estimator_ = GAM(
            terms=self.terms,
            distribution=dist_name,
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

        dist_name = self._dist_name

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
        from skbase.utils.dependencies import _check_soft_dependencies

        # if pygam isn't installed, return a marker so tests know to skip
        if not _check_soft_dependencies("pygam", severity="none"):
            return {"distribution": "runtests-no-pygam"}
        params = [
            {"distribution": "normal", "terms": "auto"},
            {"distribution": "poisson", "terms": "auto", "link": "log"},
            {"distribution": "gamma", "terms": "auto", "link": "log"},
            {
                "distribution": "normal",
                "terms": "auto",
                "max_iter": 50,
                "fit_intercept": False,
            },
            {"distribution": "poisson", "link": "identity", "max_iter": 50},
            {"distribution": "gamma", "link": "inverse", "tol": 1e-3},
        ]
        return params
