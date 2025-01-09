"""Bayesian Conjugate Linear Regression Estimator."""

__author__ = ["meraldoantonio"]

import numpy as np
import pandas as pd

from skpro.distributions import Normal
from skpro.regression.base import BaseProbaRegressor


class BayesianConjugateLinearRegressor(BaseProbaRegressor):
    """
    Bayesian probabilistic estimator for linear regression.

    This estimator models the relationship between features `X` and target `t` using
    a Bayesian linear regression framework with conjugate priors.
    Specifically, the prior and posterior of the coefficients (`w`) are
    both  multivariate Gaussians.

    Model Assumptions
    -----------------
    - **Prior for coefficients (`w`)**:
      Prior for coefficients `w` (including intercept) follow a multivariate Gaussian:
      ```
      w ~ N(coefs_prior_mu, coefs_prior_cov)
      ```
      where:
      - `coefs_prior_mu`: Mean vector of the prior distribution for `w`.
      - `coefs_prior_cov`: Covariance matrix of the prior distribution for `w`.

    - **Likelihood**:
      The likelihood of target `t` given features `X` and coefficients `w` is Gaussian:
      ```
      t | X, w ~ N(X @ w, sigma^2)
      ```
      where:
      - `sigma^2 = 1 / noise_precision`: Variance of the noise in the data.
      - `noise_precision`: Known precision (inverse variance) of the noise.

    - **Posterior for coefficients (`w`)**:
      Using Bayesian conjugacy, posterior of `w` remains Gaussian:
      ```
      w | X, t ~ N(coefs_posterior_mu, coefs_posterior_cov)
      ```
      with:
      ```
      coefs_posterior_cov = (coefs_prior_precision + noise_precision * X.T @ X)^(-1)
      coefs_posterior_mu = coefs_posterior_cov @ (
          coefs_prior_precision @ coefs_prior_mu + noise_precision * X.T @ y
      )
      ```
      where:
      - `coefs_prior_precision = inv(coefs_prior_cov)`.

    - **Predictive distribution**:
      For a new observation `x`, the predictive distribution of `y` is:
      ```
      y_pred | x ~ N(x.T @ coefs_posterior_mu, pred_var)
      ```
      where:
      ```
      pred_var = x.T @ coefs_posterior_cov @ x + 1 / noise_precision
      ```

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
    ...     coefs_prior_mu=None,
    ...     coefs_prior_cov=np.eye(X_train.shape[1]),
    ...     noise_precision=1.0,
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
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, coefs_prior_cov, coefs_prior_mu=None, noise_precision=1):
        """Initialize regressor by providing coefficent priors and noise precision.

        Parameters
        ----------
        coefs_prior_cov : 2D np.ndarray, required
            Covariance matrix of the prior for intercept and coefficients.
            If list of lists, will be converted to a 2D np.array.
            Must be positive-definite.

        coefs_prior_mu : np.ndarray column vector, optional
            Mean vector of the prior for intercept and coefficients.
            The zeroth element of the vector is the prior for the intercept.
            If not provided, assumed to be a column vector of zeroes.

        noise_precision : float
            Known precision of the Gaussian likelihood noise (beta)
            This is the inverse of the noise variance.
        """
        if coefs_prior_cov is None:
            raise ValueError("`coefs_prior_cov` must be provided.")
        else:
            self.coefs_prior_cov = coefs_prior_cov

        # Set coefs_prior_mu to a zero vector if not provided
        if coefs_prior_mu is None:
            self.coefs_prior_mu = np.zeros((self.coefs_prior_cov.shape[0], 1))
        # Ensure coefs_prior_mu is a column vector
        elif coefs_prior_mu.ndim != 2 or coefs_prior_mu.shape[1] != 1:
            raise ValueError(
                "coefs_prior_mu must be a column vector with shape (n_features, 1)."
            )
        else:
            self.coefs_prior_mu = coefs_prior_mu

        # Validate dimensions of coefs_prior_mu and coefs_prior_cov
        if self.coefs_prior_mu.shape[0] != self.coefs_prior_cov.shape[0]:
            raise ValueError(
                "Dimensionality of `coefs_prior_mu` and `coefs_prior_cov` must match."
            )

        self.noise_precision = noise_precision

        super().__init__()

    def _fit(self, X, y):
        """Fit the Bayesian Linear Regressor to the observed data.

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
        self._y_cols = y.columns

        # Construct the prior mean and covariance
        self._coefs_prior_mu = self.coefs_prior_mu
        self._coefs_prior_cov = self.coefs_prior_cov
        self._coefs_prior_precision = np.linalg.inv(self._coefs_prior_cov)

        # Perform Bayesian inference
        (
            self._coefs_posterior_mu,
            self._coefs_posterior_cov,
        ) = self._perform_bayesian_inference(
            X, y, self._coefs_prior_mu, self._coefs_prior_cov
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
        idx = X.index
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Predictive mean: X * posterior_mu
        pred_mu = X @ self._coefs_posterior_mu

        # Compute predictive variance for each data point
        pred_var_all_x_i = []
        for i in range(X.shape[0]):
            x_i = X[i, :].reshape(1, -1)
            pred_var_x_i = (
                x_i @ self._coefs_posterior_cov @ x_i.T + 1 / self.noise_precision
            )
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
            columns=self._y_cols,
            index=idx,
        )
        return self._y_pred

    def _perform_bayesian_inference(self, X, y, coefs_prior_mu, coefs_prior_cov):
        """Perform Bayesian inference for linear regression.

        Obtains the posterior distribution using normal conjugacy formula.

        Parameters
        ----------
        X : pandas DataFrame
            Feature matrix (n_samples, n_features).
        y : pandas Series
            Observed target vector (n_samples,).
        coefs_prior_mu : np.ndarray
            Mean vector of the prior Normal distribution for coefficients.
        coefs_prior_cov : np.ndarray
            Covariance matrix of the prior Normal distribution for coefficients.

        Returns
        -------
        coefs_posterior_mu : np.ndarray
            Mean vector of the posterior Normal distribution for coefficients.
        coefs_posterior_cov : np.ndarray
            Covariance matrix of the posterior Normal distribution for coefficients.
        """
        X = np.array(X)
        y = np.array(y)

        # Compute prior precision from prior covariance
        coefs_prior_precision = np.linalg.inv(coefs_prior_cov)

        # Compute posterior precision and covariance
        coefs_posterior_precision = coefs_prior_precision + self.noise_precision * (
            X.T @ X
        )
        coefs_posterior_cov = np.linalg.inv(coefs_posterior_precision)
        coefs_posterior_mu = coefs_posterior_cov @ (
            coefs_prior_precision @ coefs_prior_mu + self.noise_precision * X.T @ y
        )

        return coefs_posterior_mu, coefs_posterior_cov

    def update(self, X, y):
        """Update the posterior with new data.

        Parameters
        ----------
        X : pandas DataFrame
            New feature matrix.
        y : pandas Series or DataFrame
            New target vector.

        Returns
        -------
        self : reference to self
        """
        # Ensure y is a DataFrame
        if isinstance(y, pd.Series):
            y = y.to_frame(name="y_train")

        (
            self._coefs_posterior_mu,
            self._coefs_posterior_cov,
        ) = self._perform_bayesian_inference(
            X, y, self._coefs_posterior_mu, self._coefs_posterior_cov
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
            "coefs_prior_mu": np.zeros(10).reshape(-1, 1),
            "coefs_prior_cov": np.eye(10),
            "noise_precision": 1.0,
        }

        params2 = {
            "coefs_prior_mu": np.zeros(10).reshape(-1, 1),
            "coefs_prior_cov": np.eye(10),
            "noise_precision": 0.5,
        }

        return [params1, params2]
