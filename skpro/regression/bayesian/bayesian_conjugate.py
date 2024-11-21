"""Bayesian Linear Regression Estimator for Probabilistic Regression."""

__author__ = ["meraldoantonio"]

import numpy as np

from skpro.distributions import Normal
from skpro.regression.base import BaseProbaRegressor


class BayesianConjugateLinearRegressor(BaseProbaRegressor):
    """Bayesian probabilistic estimator for linear regression.

    This estimator uses a Normal conjugate prior for the coefficients.
    It assumes a known noise variance for the Gaussian likelihood.
    Upon inference, it returns a Normal distribution for the posterior
    as well as for predictions.
    """

    _tags = {
        "authors": ["meraldoantonio"],
        "python_dependencies": ["scipy", "matplotlib"],
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_Series_Table",
    }

    def __init__(self, prior_mean=None, prior_cov=None, noise_variance=1.0, prior=None):
        """Initialize the Bayesian Linear Regressor with priors.

        Parameters
        ----------
        prior_mean : ndarray, optional
            Mean vector for the Normal prior for coefficients. Default is None.
        prior_cov : ndarray, optional
            Covariance matrix for the Normal prior for coefficients. Default is None.
        noise_variance : float, optional
            Known variance of the Gaussian likelihood. Default is 1.0.
        prior : Normal, optional
            An existing Normal distribution prior. Default is None.

        Raises
        ------
        ValueError
            If neither (prior_mean and prior_cov) nor prior are provided.
        TypeError
            If the provided prior is not an instance of Normal.
        """
        if prior is None:
            if prior_mean is None or prior_cov is None:
                raise ValueError(
                    "Must provide either (prior_mean and prior_cov) or prior."
                )
            self.prior_mean = np.array(prior_mean)
            self.prior_cov = np.array(prior_cov)
            self.prior = Normal(mean=prior_mean, cov=prior_cov)
        else:
            if not isinstance(prior, Normal):
                raise TypeError("Prior must be an instance of Normal.")
            self.prior = prior
            self.prior_mean = prior.mean
            self.prior_cov = prior.cov

        self.noise_variance = noise_variance

        super().__init__()

    def _fit(self, X, y):
        """Fit the Bayesian Linear Regressor to the observed data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix (n_samples, n_features).
        y : pandas Series
            Target vector (n_samples,).

        Returns
        -------
        self : reference to self
        """
        self._posterior = self._perform_bayesian_inference(self.prior, X, y)
        return self

    def _predict_proba(self, X):
        """Predict the distribution over outputs for given features.

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix (n_samples, n_features).

        Returns
        -------
        y_pred : Normal
            Predicted Normal distribution for outputs.
        """
        mean_pred = X @ self._posterior.mean
        cov_pred = X @ self._posterior.cov @ X.T + self.noise_variance * np.eye(len(X))
        return Normal(mean=mean_pred, cov=cov_pred)

    def _perform_bayesian_inference(self, prior, X, y):
        """Perform Bayesian inference for linear regression.

        Obtains the posterior distribution using normal conjugacy
        formula

        Parameters
        ----------
        prior : Normal
            The prior Normal distribution.
        X : pandas DataFrame
            Feature matrix (n_samples, n_features).
        y : pandas Series
            Observed target vector (n_samples,).

        Returns
        -------
        posterior : Normal
            Posterior Normal distribution with updated parameters.
        """
        X = np.array(X)
        y = np.array(y)

        # Prior mean and covariance
        prior_mean = prior.mean
        prior_cov = prior.cov

        # Compute posterior parameters
        posterior_cov = np.linalg.inv(
            np.linalg.inv(prior_cov) + (X.T @ X) / self.noise_variance
        )
        posterior_mean = posterior_cov @ (
            np.linalg.inv(prior_cov) @ prior_mean + (X.T @ y) / self.noise_variance
        )

        return Normal(mean=posterior_mean, cov=posterior_cov)

    def update(self, X, y):
        """Update the posterior with new data.

        Parameters
        ----------
        X : pandas DataFrame
            New feature matrix.
        y : pandas Series
            New target vector.

        Returns
        -------
        self : reference to self
        """
        # in the update, the existing posterior serves as prior
        self._posterior = self._perform_bayesian_inference(self._posterior, X, y)
        return self

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
        """
        params1 = {
            "prior_mean": [0, 0],
            "prior_cov": [[1, 0], [0, 1]],
            "noise_variance": 1.0,
        }
        params2 = {
            "prior_mean": [0.5, 0.5],
            "prior_cov": [[2, 0.5], [0.5, 2]],
            "noise_variance": 0.5,
        }

        return [params1, params2]
