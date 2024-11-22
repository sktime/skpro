"""Bayesian Conjugate Linear Regression Estimator."""

__author__ = ["meraldoantonio"]

import numpy as np
import pandas as pd

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
            Example

        Example
        -------
        >>> from skpro.regression.bayesian.bayesian_conjugate import (
        ...     BayesianConjugateLinearRegressor,
        ... )  # doctest: +SKIP
        >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
        >>> from sklearn.model_selection import train_test_split  # doctest: +SKIP
        >>> import numpy as np  # doctest: +SKIP

        >>> # Load dataset
        >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)  # doctest: +SKIP

        >>> # Calculate prior mean and covariance
        >>> n_features = X_train.shape[1]  # doctest: +SKIP
        >>> prior_mean = np.zeros(n_features)  # Init. prior mu as 0's # doctest: +SKIP
        >>> prior_cov = np.eye(n_features) * 10  # Diagonal covariance # doctest: +SKIP

        >>> # Initialize Bayesian Linear Regressor with prior
        >>> bayes_model = BayesianConjugateLinearRegressor(
        ...     prior_mean=prior_mean, prior_cov=prior_cov, noise_variance=1.0
        ... )  # doctest: +SKIP

        >>> # Fit the model
        >>> bayes_model.fit(X_train, y_train)  # doctest: +SKIP

        >>> # Predict probabilities (returns a distribution)
        >>> y_test_pred_proba = bayes_model.predict_proba(X_test)  # doctest: +SKIP

        >>> # Predict point estimates (mean of the predicted distribution)
        >>> y_test_pred = bayes_model.predict(X_test)  # doctest: +SKIP

        """
        if prior is None:
            if prior_mean is None or prior_cov is None:
                raise ValueError(
                    "Must provide either (prior_mean and prior_cov) or prior."
                )
            self.prior_mean = np.array(prior_mean)
            self.prior_sigma = np.sqrt(np.array(prior_cov).diagonal())  # Convert to std
            self.prior_cov = np.array(prior_cov)
            self.prior = Normal(mu=self.prior_mean, sigma=self.prior_sigma)
        else:
            if not isinstance(prior, Normal):
                raise TypeError(
                    "Prior must be an instance of skpro Normal distribution"
                )
            self.prior = prior
            self.prior_mean = prior.mu
            self.prior_sigma = prior.sigma
            self.prior_cov = np.diag(self.prior_sigma**2)

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
        if isinstance(X, pd.DataFrame):
            X = X.values
        mean_pred = X @ self._posterior.mu
        cov_pred = X @ np.diag(self._posterior.sigma**2) @ X.T
        sigma_pred = np.sqrt(np.diag(cov_pred) + self.noise_variance)

        return Normal(mu=mean_pred, sigma=sigma_pred)

    def _perform_bayesian_inference(self, prior, X, y):
        """Perform Bayesian inference for linear regression.

        Obtains the posterior distribution using normal conjugacy formula.

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
        prior_mean = prior.mu
        prior_cov = np.diag(prior.sigma**2)  # Convert std to covariance

        # Compute posterior parameters
        posterior_cov = np.linalg.inv(
            np.linalg.inv(prior_cov) + (X.T @ X) / self.noise_variance
        )
        posterior_mean = posterior_cov @ (
            np.linalg.inv(prior_cov) @ prior_mean + (X.T @ y) / self.noise_variance
        )
        posterior_sigma = np.sqrt(np.diag(posterior_cov))
        self._posterior_mean = posterior_mean
        self._posterior_sigma = posterior_sigma
        self._posterior_cov = posterior_cov

        return Normal(mu=posterior_mean, sigma=posterior_sigma)

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
        self._posterior_mean = self._posterior.mean
        self._posterior_sigma = self._posterior.sigma
        self._posterior_cov = np.diag(self._posterior_sigma**2)

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
