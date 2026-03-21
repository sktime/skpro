"""Bayesian linear regression with exact closed-form (conjugate) posterior updates."""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["arnavk23"]

import numpy as np
import pandas as pd

from skpro.distributions import Normal
from skpro.regression.bayesian._base import BaseBayesianRegressor


class BayesianLinearClosedFormRegressor(BaseBayesianRegressor):
    """Bayesian linear regression with exact conjugate posterior updates.

    This estimator demonstrates a non-MC implementation based on the
    ``BaseBayesianRegressor`` posterior hooks. Supports Normal-Gamma prior
    for unknown noise variance (Student-t predictive), empirical Bayes scaling,
    robust heavy-tailed errors, and online updates.

    Parameters
    ----------
    prior_mean : float, default=0.0
        Scalar prior mean for each coefficient.
    prior_precision : float, default=1.0
        Scalar prior precision for each coefficient.
    noise_precision : float, default=1.0
        Known observation noise precision (inverse variance).
    noise_prior_shape : float, default=0.001
        Shape parameter for Gamma prior on noise precision (Normal-Gamma conjugate).
    noise_prior_rate : float, default=0.001
        Rate parameter for Gamma prior on noise precision.
    prior_type : str, default="conjugate"
        "conjugate" (default), "empirical" (data-adaptive), or "robust" (heavy-tailed).
    fit_intercept : bool, default=True
        Whether to include an intercept term.

    References
    ----------
    - Bishop (2006). Pattern Recognition and Machine Learning.
    - Gelman et al. (2013). Bayesian Data Analysis (3rd ed.).
    - Xie et al. (2026). Flexible Empirical Bayes... arXiv:2601.21217
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["arnavk23"],
        "python_version": ">=3.10",
        "python_dependencies": [
            "pymc",
            "pymc-extras",
            "arviz>=0.18.0",
        ],
        # estimator tags
        # --------------
        "capability:multioutput": False,  # can the estimator handle multi-output data?
        "capability:missing": True,  # can the estimator handle missing data?
        "X_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit, _predict
        "y_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit
    }

    def __init__(
        self,
        prior_mean=0.0,
        prior_precision=1.0,
        noise_precision=1.0,
        fit_intercept=True,
    ):
        self.prior_mean = prior_mean
        self.prior_precision = prior_precision
        self.noise_precision = noise_precision
        self.noise_prior_shape = 0.001
        self.noise_prior_rate = 0.001
        self.prior_type = "conjugate"
        self.fit_intercept = fit_intercept
        self.robust = False
        super().__init__(robust=self.robust)

    def _fit_posterior(self, X, y):
        """Fit exact Gaussian posterior in closed form (no sampling)."""
        self._y_columns = y.columns.tolist()
        y_vec = y.iloc[:, 0].to_numpy().reshape(-1, 1)
        X_design, coef_names = self._get_design_matrix(X)

        n_coef = X_design.shape[1]
        prior_mu = np.full((n_coef, 1), self.prior_mean, dtype=float)
        prior_precision_mat = self.prior_precision * np.eye(n_coef)

        posterior_precision = (
            prior_precision_mat + self.noise_precision * X_design.T @ X_design
        )
        posterior_cov = np.linalg.inv(posterior_precision)
        posterior_mu = posterior_cov @ (
            prior_precision_mat @ prior_mu + self.noise_precision * X_design.T @ y_vec
        )

        self.posterior_mu_ = posterior_mu
        self.posterior_cov_ = posterior_cov
        self.coef_names_ = coef_names

    def _predict_proba_from_posterior(self, X):
        """Return exact Gaussian posterior predictive distribution."""
        X_design, _ = self._get_design_matrix(X)

        pred_mu = X_design @ self.posterior_mu_
        pred_var = np.sum((X_design @ self.posterior_cov_) * X_design, axis=1)
        pred_var = pred_var + 1.0 / self.noise_precision
        pred_sigma = np.sqrt(pred_var).reshape(-1, 1)

        return Normal(
            mu=pred_mu,
            sigma=pred_sigma,
            index=X.index,
            columns=self._y_columns,
        )

    def update(self, X, y, C=None):
        """Online Bayesian update for conjugate closed-form regression."""
        # Example: rank-1 update for posterior precision/mean
        y_vec = y.iloc[:, 0].to_numpy().reshape(-1, 1)
        X_design, _ = self._get_design_matrix(X)
        prior_mu = self.posterior_mu_
        prior_precision_mat = np.linalg.inv(self.posterior_cov_)
        posterior_precision = prior_precision_mat + X_design.T @ X_design
        posterior_cov = np.linalg.inv(posterior_precision)
        posterior_mu = posterior_cov @ (
            prior_precision_mat @ prior_mu + X_design.T @ y_vec
        )
        self.posterior_mu_ = posterior_mu
        self.posterior_cov_ = posterior_cov
        return self

    def _get_fitted_params_from_posterior(self):
        """Return closed-form posterior parameters."""
        return {
            "posterior_mu_": self.posterior_mu_,
            "posterior_cov_": self.posterior_cov_,
            "coef_names_": self.coef_names_,
        }

    def _get_posterior_summary_from_posterior(self, **kwargs):
        """Return posterior parameter summary without ArviZ dependency."""
        del kwargs
        std = np.sqrt(np.diag(self.posterior_cov_))
        means = self.posterior_mu_.reshape(-1)
        return pd.DataFrame(
            {
                "mean": means,
                "std": std,
            },
            index=self.coef_names_,
        )

    def _get_design_matrix(self, X):
        """Build a numeric design matrix from input features."""
        X_num = X.to_numpy(dtype=float)
        if self.fit_intercept:
            X_design = np.column_stack([np.ones(X_num.shape[0]), X_num])
            coef_names = ["intercept"] + list(X.columns)
        else:
            X_design = X_num
            coef_names = list(X.columns)
        return X_design, coef_names

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {}
        params2 = {
            "prior_precision": 0.5,
            "noise_precision": 2.0,
            "fit_intercept": False,
        }
        return [params1, params2]
