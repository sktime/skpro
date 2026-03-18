"""Bayesian Conjugate GLM Regressor with Gaussian likelihood and conjugate priors."""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["arnavk23"]

import numpy as np

from skpro.distributions import Normal
from skpro.regression.base import BaseProbaRegressor


class BayesianConjugateGLMRegressor(BaseProbaRegressor):
    """
    Bayesian GLM with Gaussian likelihood and conjugate priors.

    This estimator models the relationship between features `X` and target `t` using
    a Bayesian GLM framework with conjugate priors (multivariate normal).
    Only Gaussian link is supported (conjugate case).

    Parameters
    ----------
    coefs_prior_cov : 2D np.ndarray, required
        Covariance matrix of the prior for intercept and coefficients.
        Must be positive-definite.
    coefs_prior_mu : np.ndarray column vector, optional
        Mean vector of the prior for intercept and coefficients.
        If not provided, assumed to be a column vector of zeroes.
    noise_precision : float
        Known precision of the Gaussian likelihood noise (inverse variance).
    add_constant : bool, default=True
        Whether to add intercept column to X.
    """

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return valid test parameters for BayesianConjugateGLMRegressor.

        Returns
        -------
        list of dict
            Each dict contains parameters for a test instance.
        """
        import numpy as np

        n_features = 10
        # First parameter set: add_constant=True (11 coefs)
        n_coefs1 = n_features + 1
        params1 = {
            "add_constant": True,
            "coefs_prior_mu": np.zeros((n_coefs1, 1)),
            "coefs_prior_cov": np.eye(n_coefs1),
            "noise_precision": 1.0,
        }
        # Second parameter set: add_constant=False (10 coefs)
        n_coefs2 = n_features
        params2 = {
            "add_constant": False,
            "coefs_prior_mu": np.ones((n_coefs2, 1)),
            "coefs_prior_cov": np.eye(n_coefs2) * 2,
            "noise_precision": 2.0,
        }
        return [params1, params2]

    _tags = {
        "object_type": "regressor_proba",
        "estimator_type": "regressor_proba",
        "authors": ["arnavk23"],
        "capability:multioutput": False,
        "capability:missing": True,
        "capability:update": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(
        self, coefs_prior_cov, coefs_prior_mu=None, noise_precision=1, add_constant=True
    ):
        if coefs_prior_cov is None:
            raise ValueError("`coefs_prior_cov` must be provided.")
        self.coefs_prior_cov = coefs_prior_cov
        if coefs_prior_mu is None:
            self.coefs_prior_mu = np.zeros((self.coefs_prior_cov.shape[0], 1))
        elif coefs_prior_mu.ndim != 2 or coefs_prior_mu.shape[1] != 1:
            raise ValueError(
                "coefs_prior_mu must be a column vector with shape (n_features, 1)."
            )
        else:
            self.coefs_prior_mu = coefs_prior_mu
        if self.coefs_prior_mu.shape[0] != self.coefs_prior_cov.shape[0]:
            raise ValueError(
                "Dimensionality of `coefs_prior_mu` and `coefs_prior_cov` must match."
            )
        self.noise_precision = noise_precision
        self.add_constant = add_constant
        super().__init__()

    def _fit(self, X, y):
        self._y_cols = y.columns
        X_arr = X.copy()
        if self.add_constant:
            X_arr = self._add_intercept(X_arr)
        X_arr = X_arr.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        self._coefs_prior_mu = self.coefs_prior_mu
        self._coefs_prior_cov = self.coefs_prior_cov
        self._coefs_prior_precision = np.linalg.inv(self._coefs_prior_cov)
        self._X_train = X_arr
        self._y_train = y_arr
        (
            self._coefs_posterior_mu,
            self._coefs_posterior_cov,
        ) = self._perform_bayesian_inference(
            X_arr, y_arr, self._coefs_prior_mu, self._coefs_prior_precision
        )
        return self

    def _predict_proba(self, X):
        idx = X.index
        X_arr = X.copy()
        if self.add_constant:
            X_arr = self._add_intercept(X_arr)
        X_arr = X_arr.to_numpy(dtype=float)
        pred_mu = X_arr @ self._coefs_posterior_mu
        pred_var_all_x_i = []
        for i in range(X_arr.shape[0]):
            x_i = X_arr[i, :].reshape(1, -1)
            pred_var_x_i = (
                x_i @ self._coefs_posterior_cov @ x_i.T + 1 / self.noise_precision
            )
            pred_var_all_x_i.append(pred_var_x_i.item())
        pred_var_all_x_i = np.array(pred_var_all_x_i)
        pred_sigma = np.sqrt(pred_var_all_x_i)
        mus = pred_mu.reshape(-1, 1).tolist()
        sigmas = pred_sigma.reshape(-1, 1).tolist()
        return Normal(mu=mus, sigma=sigmas, columns=self._y_cols, index=idx)

    def _perform_bayesian_inference(self, X, y, coefs_prior_mu, coefs_prior_precision):
        coefs_posterior_precision = coefs_prior_precision + self.noise_precision * (
            X.T @ X
        )
        prior_natural_param = coefs_prior_precision @ coefs_prior_mu
        posterior_natural_param = prior_natural_param + self.noise_precision * (X.T @ y)
        coefs_posterior_mu = np.linalg.solve(
            coefs_posterior_precision, posterior_natural_param
        )
        coefs_posterior_cov = np.linalg.inv(coefs_posterior_precision)
        return coefs_posterior_mu, coefs_posterior_cov

    def _add_intercept(self, X):
        if "const" not in X.columns:
            X = X.copy()
            X.insert(0, "const", 1.0)
        return X

    def _update(self, X, y):
        X_arr = X.copy()
        if self.add_constant:
            X_arr = self._add_intercept(X_arr)
        X_arr = X_arr.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        coefs_prior_precision = np.linalg.inv(self._coefs_posterior_cov)
        (
            self._coefs_posterior_mu,
            self._coefs_posterior_cov,
        ) = self._perform_bayesian_inference(
            X_arr, y_arr, self._coefs_posterior_mu, coefs_prior_precision
        )
        return self
