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

    Example
    -------
    >>> from skpro.regression.bayesian.bayesian_conjugate import (
    ...     BayesianConjugateLinearRegressor,
    ... )  # doctest: +SKIP
    >>> from skpro.distributions import Normal  # doctest: +SKIP
    >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
    >>> from sklearn.model_selection import train_test_split  # doctest: +SKIP
    >>> import numpy as np  # doctest: +SKIP

    >>> # Load dataset
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)  # doctest: +SKIP

    >>> # Define prior coefficients as a Normal distribution
    >>> n_features = X_train.shape[1]  # doctest: +SKIP
    >>> prior_coefficients = Normal(
    ...     mu=np.zeros(n_features), sigma=np.ones(n_features) * 10
    ... )  # doctest: +SKIP

    >>> # Initialize model
    >>> bayes_model = BayesianConjugateLinearRegressor(
    ...     prior_coefficients=prior_coefficients, noise_variance=1.0
    ... )  # doctest: +SKIP

    >>> # Fit the model
    >>> bayes_model.fit(X_train, y_train)  # doctest: +SKIP

    >>> # Predict probabilities (returns an skpro Normal distribution)
    >>> y_test_pred_proba = bayes_model.predict_proba(X_test)  # doctest: +SKIP

    >>> # Predict point estimates (mean of the predicted distribution)
    >>> y_test_pred = bayes_model.predict(X_test)  # doctest: +SKIP
    """

    _tags = {
        "authors": ["meraldoantonio"],
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_Series_Table",
    }

    def __init__(self, prior_coefficients, noise_variance=1.0):
        """Initialize the Bayesian Linear Regressor with prior coefficients.

        Parameters
        ----------
        prior_coefficients : Normal
            A Normal distribution instance representing the prior coefficients.
        noise_variance : float, optional
            Known variance of the Gaussian likelihood. Default is 1.0.
        """
        if not isinstance(prior_coefficients, Normal):
            raise TypeError(
                "prior_coefficients must be an instance of skpro Normal distribution"
            )
        self.prior_coefficients = prior_coefficients
        self._prior_coefficients_mu = self.prior_coefficients.mu
        self._prior_coefficients_sigma = self.prior_coefficients.sigma
        self._prior_coefficients_cov = np.diag(self._prior_coefficients_sigma**2)

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
        self._posterior_coefficients = self._perform_bayesian_inference(
            self.prior_coefficients, X, y
        )
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
        posterior_coefficients_mu = X @ self._posterior_coefficients.mu
        posterior_coefficients_cov = (
            X @ np.diag(self._posterior_coefficients.sigma**2) @ X.T
        )
        posterior_coefficients_sigma = np.sqrt(
            np.diag(posterior_coefficients_cov) + self.noise_variance
        )

        return Normal(mu=posterior_coefficients_mu, sigma=posterior_coefficients_sigma)

    def _perform_bayesian_inference(self, prior_coefficients, X, y):
        """Perform Bayesian inference for linear regression.

        Obtains the posterior distribution using normal conjugacy formula.

        Parameters
        ----------
        prior_coefficients : Normal
            The prior Normal distribution for coefficients.
        X : pandas DataFrame
            Feature matrix (n_samples, n_features).
        y : pandas Series
            Observed target vector (n_samples,).

        Returns
        -------
        posterior_coefficients : Normal
            Posterior Normal distribution with updated parameters.
        """
        X = np.array(X)
        y = np.array(y)

        # Prior parameters
        prior_coefficients_mu = prior_coefficients.mu
        prior_coefficients_cov = np.diag(prior_coefficients.sigma**2)

        # Compute posterior parameters
        posterior_coefficients_cov = np.linalg.inv(
            np.linalg.inv(prior_coefficients_cov) + (X.T @ X) / self.noise_variance
        )
        posterior_coefficients_mu = posterior_coefficients_cov @ (
            np.linalg.inv(prior_coefficients_cov) @ prior_coefficients_mu
            + (X.T @ y) / self.noise_variance
        )
        posterior_coefficients_sigma = np.sqrt(np.diag(posterior_coefficients_cov))

        # Save posterior attributes
        self._posterior_coefficients_mu = posterior_coefficients_mu
        self._posterior_coefficients_sigma = posterior_coefficients_sigma
        self._posterior_coefficients_cov = posterior_coefficients_cov

        return Normal(mu=posterior_coefficients_mu, sigma=posterior_coefficients_sigma)

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
        # Update posterior by treating the current posterior as the new prior
        self._posterior_coefficients = self._perform_bayesian_inference(
            self._posterior_coefficients, X, y
        )
        self._posterior_coefficients_mu = self._posterior_coefficients.mu
        self._posterior_coefficients_sigma = self._posterior_coefficients.sigma
        self._posterior_coefficients_cov = np.diag(
            self._posterior_coefficients_sigma**2
        )

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
        n_features = 10
        params1 = {
            "prior_coefficients": Normal(
                mu=np.zeros(n_features), sigma=np.ones(n_features) * 5
            ),
            "noise_variance": 1.0,
        }
        params2 = {
            "prior_coefficients": Normal(
                mu=np.random.uniform(0, 1, n_features),
                sigma=np.random.uniform(1, 3, n_features),
            ),
            "noise_variance": 2.0,
        }

        return [params1, params2]
