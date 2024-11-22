"""Bayesian Conjugate Linear Regression Estimator."""

__author__ = ["meraldoantonio"]

import numpy as np
import pandas as pd

from skpro.distributions import Normal
from skpro.regression.base import BaseProbaRegressor


class BayesianConjugateLinearRegressor(BaseProbaRegressor):
    """Bayesian probabilistic estimator for linear regression.

    This estimator uses a multivariate Normal conjugate prior for the coefficients,
    with the prior mean fixed at zero and precision controlled by a scalar alpha.
    It assumes a known noise precision beta for the Gaussian likelihood.
    For inference, it returns a multivariate Normal distribution as the posterior.
    For prediction, it returns a series of univariate Normals for each data point.

    Example
    -------
    >>> from skpro.regression.bayesian.bayesian_conjugate import (
    ...     BayesianConjugateLinearRegressor,
    ... )  # doctest: +SKIP
    >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
    >>> from sklearn.model_selection import train_test_split  # doctest: +SKIP

    >>> # Load dataset
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)  # doctest: +SKIP

    >>> # Center the training data
    >>> X_train -= X_train.mean(axis=0)  # doctest: +SKIP

    >>> # Initialize model
    >>> bayes_model = BayesianConjugateLinearRegressor(
    ...     alpha=10.0, beta=1.0, n_features=X_train.shape[1]
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

    def __init__(self, alpha, beta, n_features):
        """Initialize the Bayesian Linear Regressor.

        Parameters
        ----------
        alpha : float
            Scalar precision parameter of the coefficients normal prior.
        beta : float
            Known precision of the Gaussian likelihood.
        n_features : int
            Number of features (coefficients) in the linear model.
        """
        self.alpha = alpha
        self.beta = beta
        self.n_features = n_features

        # Construct the prior mean, covariance and precision
        self._prior_mu = np.zeros(self.n_features)
        self._prior_cov = np.eye(self.n_features) / self.alpha
        self._prior_precision = np.linalg.inv(self._prior_cov)

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
        # Ensure dimensions match the number of features
        assert X.shape[1] == self.n_features, (
            f"Expected {self.n_features} features, " f"but got {X.shape[1]} features."
        )

        # Ensure the data is centered
        is_centered = np.allclose(X.mean(axis=0), 0)
        if not is_centered:
            X -= X.mean(axis=0)

        # Perform Bayesian inference
        self._posterior_mu, self._posterior_cov = self._perform_bayesian_inference(X, y)
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
        idx = X.index
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Predictive mean: X * posterior_mu
        pred_mu = X @ self._posterior_mu

        # Compute predictive variance for each data point
        pred_var_all_x_i = []
        for i in range(X.shape[0]):
            x_i = X[i, :].reshape(1, -1)
            pred_var_x_i = x_i @ self._posterior_cov @ x_i.T + 1 / self.beta
            pred_var_all_x_i.append(pred_var_x_i.item())

        pred_var_all_x_i = np.array(pred_var_all_x_i)
        pred_sigma = np.sqrt(pred_var_all_x_i)  # Compute standard deviation

        self._pred_mu = pred_mu
        self._pred_sigma = pred_sigma
        self._pred_var = pred_var_all_x_i

        mus = pred_mu.reshape(-1, 1).tolist()  # Convert to list of lists
        sigmas = pred_sigma.reshape(-1, 1).tolist()  # Convert to list of lists

        # Return results as skpro Normal distribution
        self._y_pred = Normal(
            mu=mus,
            sigma=sigmas,
            columns=["y_pred"],
            index=idx,
        )
        return self._y_pred

    def _perform_bayesian_inference(self, X, y):
        """Perform Bayesian inference for linear regression.

        Obtains the posterior distribution using normal conjugacy formula.

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix (n_samples, n_features).
        y : pandas Series
            Observed target vector (n_samples,).

        Returns
        -------
        posterior_mu : np.ndarray
            Mean vector of the posterior Normal distribution for coefficients.
        posterior_cov : np.ndarray
            Covariance matrix of the posterior Normal distribution for coefficients.
        """
        X = np.array(X)
        y = np.array(y)

        # Compute posterior precision and covariance
        posterior_precision = self._prior_precision + self.beta * (X.T @ X)
        posterior_cov = np.linalg.inv(posterior_precision)

        # Compute posterior mean
        posterior_mu = posterior_cov @ (
            self._prior_precision @ self._prior_mu + self.beta * X.T @ y
        )

        return posterior_mu, posterior_cov

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
        self._posterior_mu, self._posterior_cov = self._perform_bayesian_inference(
            X, y, self._posterior_mu, self._posterior_cov
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
        params1 = {
            "alpha": 10.0,
            "beta": 1.0,
            "n_features": 10,
        }

        return [params1]
