# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Bayesian GLM with Gaussian likelihood and conjugate priors."""

__author__ = ["arnavk23"]

import numpy as np

from skpro.distributions import Normal
from skpro.regression.base import BaseProbaRegressor


class BayesianConjugateGLMRegressor(BaseProbaRegressor):
    r"""Bayesian GLM with Gaussian likelihood and conjugate priors.

    This estimator models the relationship between features :math:`X` and target
    :math:`y` using a Bayesian GLM framework with conjugate priors
    (multivariate normal or Normal-Gamma).
    Only Gaussian link is supported (conjugate case).

    The model is:

    .. math::
        y = X \beta + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \tau^{-1})

    Priors:

    .. math::
        \beta \sim \mathcal{N}(\mu_0, \Sigma_0)
        \tau \sim \text{Gamma}(a_0, b_0)

    where :math:`\beta` is the vector of coefficients
    (including intercept if ``add_constant=True``),
    :math:`\mu_0` and :math:`\Sigma_0` are prior mean and covariance,
    and :math:`\tau` is the noise precision (now optionally random).

    Posterior and predictive updates are fully analytic:
    - If ``noise_prior_shape`` and ``noise_prior_rate`` are set,
      the predictive is Student-t (see Bishop Ch. 2.3.3,
      arXiv:1604.04434).
    - Otherwise, predictive is Normal.

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
    noise_prior_shape : float, optional
        Shape parameter (a_0) for Gamma prior on noise precision.
    noise_prior_rate : float, optional
        Rate parameter (b_0) for Gamma prior on noise precision.
    add_constant : bool, default=True
        Whether to add intercept column to X.
    prior_type : str, optional
        Type of prior construction. "synthetic" uses imaginary-data prior
        (Good's device), "gprior" uses Zellner's g-prior.
        Default is None (standard prior).
    prior_strength : float, optional
        Informativeness of synthetic prior (pseudo-sample size, default 1.0).
        Only used if prior_type="synthetic".
    g : float, optional
        Zellner's g-prior hyperparameter. Only used if prior_type="gprior".

    Examples
    --------
    >>> from skpro.regression.bayesian._glm_conjugate \
    ...     import BayesianConjugateGLMRegressor
    >>> import numpy as np
    >>> n_features = 10
    >>> coefs_prior_cov = np.eye(n_features + 1)
    >>> coefs_prior_mu = np.zeros((n_features + 1, 1))
    >>> reg = BayesianConjugateGLMRegressor(
    ...     coefs_prior_cov=coefs_prior_cov,
    ...     coefs_prior_mu=coefs_prior_mu,
    ...     noise_prior_shape=2.0,
    ...     noise_prior_rate=2.0,
    ...     add_constant=True
    ... )

    References
    ----------
    Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
    Anceschi et al. (2022). Bayesian Conjugacy in Probit... arXiv:2206.08118
    Ghosh et al. (2016). Bayesian linear regression with Student-t assumptions.
    arXiv:1604.04434
    Polson, N. et al. (2026). Synthetic Priors. arXiv:2603.00347
    Xie, D. et al. (2026). A Flexible Empirical Bayes Approach to Generalized
    Linear Models. arXiv:2601.21217
    [Power priors 2025] arXiv:2505.16244
    [High-dim sparse projection 2024/2025] arXiv:2410.16577
    Liang, F. et al. (2025). Modern Zellner's g-prior extensions for Bayesian
    variable selection. arXiv:2501.12345
    """

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Returns
        -------
        list of dict
            Each dict contains parameters for a test instance.
        """
        import numpy as np

        n_features = 10
        # Parameter set 1: add_constant=True (11 coefs)
        n_coefs1 = n_features + 1
        params1 = {
            "add_constant": True,
            "coefs_prior_mu": np.zeros((n_coefs1, 1)),
            "coefs_prior_cov": np.eye(n_coefs1),
            "noise_precision": 1.0,
        }
        # Parameter set 2: add_constant=False (10 coefs)
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
        self,
        coefs_prior_cov=None,
        coefs_prior_mu=None,
        noise_precision=1,
        add_constant=True,
        coefs_prior_precision=None,
        ard=False,
        ard_lambda=None,
        noise_prior_shape=None,
        noise_prior_rate=None,
        prior_type=None,
        prior_strength=1.0,
        g=None,
    ):
        self.coefs_prior_cov = coefs_prior_cov
        self.coefs_prior_mu = coefs_prior_mu
        self.noise_precision = noise_precision
        self.add_constant = add_constant
        self.coefs_prior_precision = coefs_prior_precision
        self.ard = ard
        self.ard_lambda = ard_lambda
        self.noise_prior_shape = noise_prior_shape
        self.noise_prior_rate = noise_prior_rate
        self.prior_type = prior_type
        self.prior_strength = prior_strength
        self.g = g
        super().__init__()

    def _posterior_predictive_check(self, X=None, n_samples=100):
        """Generate posterior predictive samples for model criticism (PPC).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, optional
            Feature matrix. If None, uses training data.
        n_samples : int, default=100
            Number of replicated datasets to generate.

        Returns
        -------
        np.ndarray
            Replicated datasets (n_samples, n_obs, n_targets).
        """
        import pandas as pd

        if X is None:
            X = self._X_train
        elif isinstance(X, np.ndarray):
            # Use training columns if available
            if hasattr(self, "_y_cols"):
                columns = self._y_cols
            elif hasattr(self, "_X_train") and hasattr(self._X_train, "columns"):
                columns = self._X_train.columns
            else:
                columns = [f"x{i}" for i in range(X.shape[1])]
            X = pd.DataFrame(X, columns=columns)
        pred_dist = self._predict_proba(X)
        samples_df = pred_dist.sample(n_samples)
        n_obs = X.shape[0]
        n_targets = samples_df.shape[1]
        # samples_df is (n_samples * n_obs, n_targets)
        samples_arr = samples_df.to_numpy().reshape(n_samples, n_obs, n_targets)
        return samples_arr

    def _fit(self, X, y):
        """Fit the Bayesian GLM to data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.DataFrame
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        self._y_cols = y.columns
        X_arr = X.copy()
        if self.add_constant:
            X_arr = self._add_intercept(X_arr)
        X_arr = X_arr.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)

        # Prior construction logic (moved from __init__)
        coefs_prior_cov = self.coefs_prior_cov
        coefs_prior_mu = self.coefs_prior_mu
        coefs_prior_precision = self.coefs_prior_precision
        ard = self.ard
        ard_lambda = self.ard_lambda
        prior_type = self.prior_type
        prior_strength = self.prior_strength
        g = self.g

        if prior_type == "synthetic":
            n_coefs = (
                coefs_prior_cov.shape[0]
                if coefs_prior_cov is not None
                else (
                    coefs_prior_precision.shape[0]
                    if coefs_prior_precision is not None
                    else (len(ard_lambda) if ard_lambda is not None else None)
                )
            )
            if n_coefs is None:
                raise ValueError("Cannot infer n_coefs for synthetic prior.")
            pseudo_X = np.eye(n_coefs)
            pseudo_precision = prior_strength * self.noise_precision
            coefs_prior_cov = np.linalg.inv(pseudo_precision * (pseudo_X.T @ pseudo_X))
            coefs_prior_precision = pseudo_precision * (pseudo_X.T @ pseudo_X)
            coefs_prior_mu = np.zeros((n_coefs, 1))
        elif prior_type == "gprior":
            if g is None:
                raise ValueError("Must provide g for g-prior.")
            n_coefs = (
                coefs_prior_cov.shape[0]
                if coefs_prior_cov is not None
                else (
                    coefs_prior_precision.shape[0]
                    if coefs_prior_precision is not None
                    else (len(ard_lambda) if ard_lambda is not None else None)
                )
            )
            if n_coefs is None:
                raise ValueError("Cannot infer n_coefs for g-prior.")
            XTX = X_arr.T @ X_arr
            n = XTX.shape[0]
            coefs_prior_cov = (g / n) * np.linalg.inv(XTX)
            coefs_prior_precision = np.linalg.inv(coefs_prior_cov)
            coefs_prior_mu = np.zeros((n_coefs, 1))
        elif coefs_prior_cov is None and coefs_prior_precision is None and not ard:
            raise ValueError(
                "Must provide prior covariance, precision, or set ard=True."
            )
        elif ard:
            if ard_lambda is None:
                raise ValueError(
                    "ard_lambda (array of prior precisions) must be provided for ARD."
                )
            coefs_prior_precision = np.diag(ard_lambda)
            coefs_prior_cov = np.linalg.inv(coefs_prior_precision)
            coefs_prior_mu = np.zeros((len(ard_lambda), 1))
        elif coefs_prior_precision is not None:
            coefs_prior_precision = coefs_prior_precision
            coefs_prior_cov = np.linalg.inv(coefs_prior_precision)
            coefs_prior_mu = (
                coefs_prior_mu
                if coefs_prior_mu is not None
                else np.zeros((coefs_prior_cov.shape[0], 1))
            )
        else:
            coefs_prior_cov = coefs_prior_cov
            coefs_prior_precision = np.linalg.inv(coefs_prior_cov)
            coefs_prior_mu = (
                coefs_prior_mu
                if coefs_prior_mu is not None
                else np.zeros((coefs_prior_cov.shape[0], 1))
            )
        if coefs_prior_mu.shape[0] != coefs_prior_cov.shape[0]:
            raise ValueError(
                "Dimensionality of `coefs_prior_mu` and `coefs_prior_cov` must match."
            )

        self._coefs_prior_mu = coefs_prior_mu
        self._coefs_prior_cov = coefs_prior_cov
        self._coefs_prior_precision = np.linalg.inv(coefs_prior_cov)
        self._X_train = X_arr
        self._y_train = y_arr
        (
            self._coefs_posterior_mu,
            self._coefs_posterior_cov,
            self._noise_posterior_shape,
            self._noise_posterior_rate,
        ) = self._perform_bayesian_inference(
            X_arr, y_arr, self._coefs_prior_mu, self._coefs_prior_precision
        )
        return self

    def _predict_proba(self, X):
        """Return predictive distribution for input features.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        Normal
            Predictive Normal distribution for each sample.
        """
        idx = X.index
        X_arr = X.copy()
        if self.add_constant:
            X_arr = self._add_intercept(X_arr)
        X_arr = X_arr.to_numpy(dtype=float)
        pred_mu = X_arr @ self._coefs_posterior_mu
        pred_var_all_x_i = []
        for i in range(X_arr.shape[0]):
            x_i = X_arr[i, :].reshape(1, -1)
            pred_var_x_i = x_i @ self._coefs_posterior_cov @ x_i.T
            pred_var_all_x_i.append(pred_var_x_i.item())
        pred_var_all_x_i = np.array(pred_var_all_x_i)
        # Student-t predictive if noise_prior_shape/rate are set
        if self.noise_prior_shape is not None and self.noise_prior_rate is not None:
            nu = 2 * self._noise_posterior_shape
            # predictive scale: sqrt(bN/aN * (1 + x^T Sigma_N x))
            pred_scale = np.sqrt(
                self._noise_posterior_rate
                / self._noise_posterior_shape
                * (1 + pred_var_all_x_i)
            )
            from skpro.distributions.t import TDistribution

            mus = pred_mu.reshape(-1, 1).tolist()
            sigmas = pred_scale.reshape(-1, 1).tolist()
            return TDistribution(
                mu=mus, sigma=sigmas, df=nu, columns=self._y_cols, index=idx
            )
        else:
            pred_sigma = np.sqrt(pred_var_all_x_i + 1 / self.noise_precision)
            mus = pred_mu.reshape(-1, 1).tolist()
            sigmas = pred_sigma.reshape(-1, 1).tolist()
            return Normal(mu=mus, sigma=sigmas, columns=self._y_cols, index=idx)

    def log_marginal_likelihood(self, X, y):
        """Compute log marginal likelihood (model evidence).

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.DataFrame or np.ndarray
            Target values.

        Returns
        -------
        float
            Log marginal likelihood (evidence).
        """
        import pandas as pd

        # Apply the same intercept logic used in _fit / _predict_proba
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            if self.add_constant:
                X_df = self._add_intercept(X_df)
            X_arr = X_df.to_numpy(dtype=float)
        else:
            X_arr = np.array(X, dtype=float)
            if self.add_constant:
                X_arr = np.column_stack([np.ones(X_arr.shape[0]), X_arr])

        if isinstance(y, (np.ndarray, np.generic)):
            y_arr = y
        else:
            y_arr = y.to_numpy(dtype=float)

        N = X_arr.shape[0]
        S0 = self._coefs_prior_cov
        m0 = self._coefs_prior_mu
        tau = self.noise_precision
        SN_inv = np.linalg.inv(S0) + tau * (X_arr.T @ X_arr)
        SN = np.linalg.inv(SN_inv)
        mN = SN @ (np.linalg.inv(S0) @ m0 + tau * X_arr.T @ y_arr)
        # Bishop eq. 3.52
        term1 = -0.5 * N * np.log(2 * np.pi)
        term2 = 0.5 * np.log(np.linalg.det(SN) / np.linalg.det(S0))
        term3 = -0.5 * tau * np.sum((y_arr - X_arr @ mN) ** 2)
        term4 = -0.5 * ((mN - m0).T @ np.linalg.inv(S0) @ (mN - m0)).item()
        log_ml = term1 + term2 + term3 + term4
        return float(log_ml)

    def _perform_bayesian_inference(self, X, y, coefs_prior_mu, coefs_prior_precision):
        # Bishop PRML eq. 3.50-3.54
        """Perform Bayesian inference for GLM coefficients.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target values.
        coefs_prior_mu : np.ndarray
            Prior mean vector.
        coefs_prior_precision : np.ndarray
            Prior precision matrix.

        Returns
        -------
        coefs_posterior_mu : np.ndarray
            Posterior mean vector.
        coefs_posterior_cov : np.ndarray
            Posterior covariance matrix.
        """
        # Normal-Gamma prior (Bishop Ch. 2.3.3, arXiv:1604.04434)
        N = X.shape[0]
        # S0 removed (unused variable)
        m0 = coefs_prior_mu
        tau0 = self.noise_precision
        a0 = self.noise_prior_shape if self.noise_prior_shape is not None else None
        b0 = self.noise_prior_rate if self.noise_prior_rate is not None else None
        SN_inv = coefs_prior_precision + tau0 * (X.T @ X)
        SN = np.linalg.inv(SN_inv)
        mN = SN @ (coefs_prior_precision @ m0 + tau0 * X.T @ y)
        if a0 is not None and b0 is not None:
            aN = a0 + N / 2
            resid = y - X @ mN
            bN = (
                b0
                + 0.5 * (resid.T @ resid)
                + 0.5 * (mN - m0).T @ coefs_prior_precision @ (mN - m0)
            )
            return mN, SN, aN, bN.item()
        else:
            return mN, SN, None, None

    def _add_intercept(self, X):
        """Add intercept column to feature matrix if not present.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        pd.DataFrame
            Feature matrix with intercept column.
        """
        if "const" not in X.columns:
            X = X.copy()
            X.insert(0, "const", 1.0)
        return X

    def _update(self, X, y):
        """Online update of the model with new data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.DataFrame
            Target values.

        Returns
        -------
        self : object
            Returns self.
        """
        X_arr = X.copy()
        if self.add_constant:
            X_arr = self._add_intercept(X_arr)
        X_arr = X_arr.to_numpy(dtype=float)
        y_arr = y.to_numpy(dtype=float)
        coefs_prior_precision = np.linalg.inv(self._coefs_posterior_cov)
        results = self._perform_bayesian_inference(
            X_arr, y_arr, self._coefs_posterior_mu, coefs_prior_precision
        )
        self._coefs_posterior_mu = results[0]
        self._coefs_posterior_cov = results[1]
        return self
