"""Bayesian Ridge Regression estimator with evidence maximization."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["david_laid"]

import numpy as np
import pandas as pd

from skpro.distributions.normal import Normal
from skpro.regression.bayesian._base import BaseBayesianRegressor


class BayesianRidgeRegressor(BaseBayesianRegressor):
    r"""Bayesian Ridge Regression with automatic evidence maximization.

    Fits a linear model ``y = X w + ε`` with Gaussian prior on weights
    and Gaussian noise, using Type-II maximum likelihood (empirical Bayes)
    to optimize the precision hyperparameters.

    Model
    -----
    .. math::

        w \sim \mathcal{N}(0, \alpha^{-1} I)

        y | X, w \sim \mathcal{N}(X w, \beta^{-1} I)

    The hyperparameters ``α`` (weight precision) and ``β`` (noise precision)
    are optimized by maximizing the log marginal likelihood (evidence).
    The posterior over ``w`` is then available in closed form:

    .. math::

        w | X, y \sim \mathcal{N}(m_N, S_N)

        S_N = (\alpha I + \beta X^T X)^{-1}

        m_N = \beta S_N X^T y

    Parameters
    ----------
    alpha_init : float, default=1e-6
        Initial value for weight precision ``α``.
    beta_init : float, default=1e-6
        Initial value for noise precision ``β``.
    n_iter : int, default=300
        Maximum number of evidence maximization iterations.
    tol : float, default=1e-3
        Convergence tolerance on the log marginal likelihood.
    fit_intercept : bool, default=True
        Whether to fit an intercept term.

    Attributes
    ----------
    alpha_ : float
        Optimized weight precision.
    beta_ : float
        Optimized noise precision.
    coef_mean_ : np.ndarray of shape (n_features,)
        Posterior mean of the weight vector.
    coef_cov_ : np.ndarray of shape (n_features, n_features)
        Posterior covariance of the weight vector.
    intercept_ : float
        Fitted intercept (0.0 if ``fit_intercept=False``).

    Examples
    --------
    >>> from skpro.regression.bayesian._ridge import BayesianRidgeRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=42
    ... )  # doctest: +SKIP
    >>> reg = BayesianRidgeRegressor()  # doctest: +SKIP
    >>> reg.fit(X_train, y_train)  # doctest: +SKIP
    >>> dist = reg.predict_proba(X_test)  # doctest: +SKIP
    >>> dist.mean()  # doctest: +SKIP
    """

    _tags = {
        "authors": ["david_laid"],
        "maintainers": ["david_laid"],
        "capability:multioutput": False,
        "capability:missing": True,
        "capability:update": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(
        self,
        alpha_init=1e-6,
        beta_init=1e-6,
        n_iter=300,
        tol=1e-3,
        fit_intercept=True,
    ):
        self.alpha_init = alpha_init
        self.beta_init = beta_init
        self.n_iter = n_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

        super().__init__()

    # ------------------------------------------------------------------
    # Core fit / predict
    # ------------------------------------------------------------------

    def _fit(self, X, y):
        """Fit Bayesian Ridge via evidence maximization.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
        y : pd.DataFrame of shape (n_samples, 1)

        Returns
        -------
        self
        """
        self._y_cols = y.columns

        X_np = X.values.astype(np.float64)
        y_np = y.values[:, 0].astype(np.float64)

        # Centre data if fitting intercept
        if self.fit_intercept:
            self._X_mean = X_np.mean(axis=0)
            self._y_mean = y_np.mean()
            X_np = X_np - self._X_mean
            y_np = y_np - self._y_mean
        else:
            self._X_mean = np.zeros(X_np.shape[1])
            self._y_mean = 0.0

        n_samples, n_features = X_np.shape

        # Precompute X^T X and X^T y
        XtX = X_np.T @ X_np
        Xty = X_np.T @ y_np

        # Eigendecompose X^T X for stable evidence updates
        eigenvalues = np.linalg.eigvalsh(XtX)

        alpha = float(self.alpha_init)
        beta = float(self.beta_init)
        log_ml_prev = -np.inf

        for _ in range(self.n_iter):
            # Posterior covariance and mean
            S_N_inv = alpha * np.eye(n_features) + beta * XtX
            S_N = np.linalg.inv(S_N_inv)
            m_N = beta * S_N @ Xty

            # Effective number of well-determined parameters
            gamma = np.sum(beta * eigenvalues / (alpha + beta * eigenvalues))

            # Update hyperparameters
            alpha = float(gamma / (m_N @ m_N))
            residuals = y_np - X_np @ m_N
            beta = float(
                (n_samples - gamma) / (residuals @ residuals)
            )

            # Log marginal likelihood for convergence check
            log_ml = 0.5 * (
                n_features * np.log(alpha)
                + n_samples * np.log(beta)
                - beta * (residuals @ residuals)
                - alpha * (m_N @ m_N)
                - np.linalg.slogdet(S_N_inv)[1]
                - n_samples * np.log(2 * np.pi)
            )

            if np.abs(log_ml - log_ml_prev) < self.tol:
                break
            log_ml_prev = log_ml

        # Store fitted parameters
        self.alpha_ = alpha
        self.beta_ = beta
        self.coef_mean_ = m_N
        self.coef_cov_ = S_N
        self.intercept_ = self._y_mean - self._X_mean @ m_N

        # Store for potential sequential updates
        self._XtX = XtX
        self._Xty = Xty
        self._n_samples = n_samples
        self._eigenvalues = eigenvalues

        return self

    def _predict_proba(self, X):
        """Return posterior predictive distribution.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)

        Returns
        -------
        dist : Normal
            Posterior predictive Normal distribution.
        """
        idx = X.index
        X_np = X.values.astype(np.float64)

        # Predictive mean
        pred_mean = X_np @ self.coef_mean_ + self.intercept_

        # Predictive variance = noise variance + model uncertainty
        # var(y*) = 1/beta + x^T S_N x
        X_centred = X_np - self._X_mean
        model_var = np.sum((X_centred @ self.coef_cov_) * X_centred, axis=1)
        pred_var = 1.0 / self.beta_ + model_var
        pred_std = np.sqrt(np.maximum(pred_var, 1e-12))

        return Normal(
            mu=pred_mean.reshape(-1, 1).tolist(),
            sigma=pred_std.reshape(-1, 1).tolist(),
            columns=self._y_cols,
            index=idx,
        )

    # ------------------------------------------------------------------
    # Bayesian API (BaseBayesianRegressor interface)
    # ------------------------------------------------------------------

    def _get_prior_params(self):
        """Return prior distributions over model parameters."""
        n_features = self.coef_mean_.shape[0] if hasattr(self, "coef_mean_") else 1
        alpha = self.alpha_ if hasattr(self, "alpha_") else self.alpha_init
        beta = self.beta_ if hasattr(self, "beta_") else self.beta_init

        prior_std = (1.0 / alpha) ** 0.5
        return {
            "coefficients": Normal(
                mu=0.0,
                sigma=prior_std,
            ),
            "noise_std": Normal(
                mu=0.0,
                sigma=(1.0 / beta) ** 0.5,
            ),
        }

    def _get_posterior_params(self):
        """Return posterior distributions over model parameters."""
        post_std = np.sqrt(np.diag(self.coef_cov_))

        return {
            "coefficients": Normal(
                mu=self.coef_mean_.tolist(),
                sigma=post_std.tolist(),
            ),
        }

    def _update(self, X, y, C=None):
        """Sequential Bayesian update with new data.

        Uses current posterior as the new prior and re-derives the
        posterior incorporating the new observations.

        Parameters
        ----------
        X : pd.DataFrame
        y : pd.DataFrame

        Returns
        -------
        self
        """
        X_np = X.values.astype(np.float64)
        y_np = y.values[:, 0].astype(np.float64)

        if self.fit_intercept:
            X_np = X_np - self._X_mean
            y_np = y_np - self._y_mean

        n_new = X_np.shape[0]

        # Accumulate sufficient statistics
        self._XtX += X_np.T @ X_np
        self._Xty += X_np.T @ y_np
        self._n_samples += n_new

        # Re-derive posterior with current alpha/beta
        n_features = X_np.shape[1]
        S_N_inv = self.alpha_ * np.eye(n_features) + self.beta_ * self._XtX
        S_N = np.linalg.inv(S_N_inv)
        m_N = self.beta_ * S_N @ self._Xty

        self.coef_mean_ = m_N
        self.coef_cov_ = S_N
        self.intercept_ = self._y_mean - self._X_mean @ m_N

        return self

    # ------------------------------------------------------------------
    # Test params for skpro test suite
    # ------------------------------------------------------------------

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"

        Returns
        -------
        params : list of dict
        """
        params1 = {
            "alpha_init": 1e-6,
            "beta_init": 1e-6,
            "n_iter": 100,
        }
        params2 = {
            "alpha_init": 1.0,
            "beta_init": 1.0,
            "n_iter": 50,
            "fit_intercept": False,
        }
        return [params1, params2]
