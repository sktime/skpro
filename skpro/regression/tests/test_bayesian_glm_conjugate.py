"""Tests for BayesianConjugateGLMRegressor."""

import numpy as np
import pandas as pd
import pytest

from skpro.regression.bayesian._glm_conjugate import BayesianConjugateGLMRegressor


def test_bayesian_conjugate_glm_regressor_basic():
    # Create synthetic data
    X = pd.DataFrame(np.random.randn(20, 2), columns=["feat1", "feat2"])
    y = pd.DataFrame(np.random.randn(20, 1), columns=["target"])

    # Minimal valid parameters
    coefs_prior_cov = np.eye(3)  # 2 features + intercept
    coefs_prior_mu = np.zeros((3, 1))
    noise_precision = 1.0
    add_constant = True

    est = BayesianConjugateGLMRegressor(
        coefs_prior_cov=coefs_prior_cov,
        coefs_prior_mu=coefs_prior_mu,
        noise_precision=noise_precision,
        add_constant=add_constant,
    )
    est.fit(X, y)
    y_pred = est.predict(X)
    y_pred_proba = est.predict_proba(X)

    assert y_pred.shape == y.shape
    assert hasattr(y_pred_proba, "mu")
    assert hasattr(y_pred_proba, "sigma")
    assert len(y_pred_proba.mu) == y.shape[0]
    assert len(y_pred_proba.sigma) == y.shape[0]
