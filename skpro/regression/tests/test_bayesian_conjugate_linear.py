"""Tests for BayesianConjugateLinearRegressor online updates."""

import numpy as np
import pandas as pd
import pytest

from skpro.regression.bayesian import BayesianConjugateLinearRegressor
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(BayesianConjugateLinearRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_bayesian_conjugate_update_matches_batch_fit():
    """Sequential posterior update should match one-shot batch posterior."""
    rng = np.random.default_rng(42)
    n_features = 3

    X1 = pd.DataFrame(rng.normal(size=(25, n_features)))
    X2 = pd.DataFrame(rng.normal(size=(20, n_features)))

    true_w = np.array([[1.2], [-0.7], [0.3]])
    y1 = pd.DataFrame(X1.values @ true_w + 0.3 * rng.normal(size=(25, 1)))
    y2 = pd.DataFrame(X2.values @ true_w + 0.3 * rng.normal(size=(20, 1)))

    prior_mu = np.zeros((n_features, 1))
    prior_cov = np.eye(n_features)

    model_sequential = BayesianConjugateLinearRegressor(
        coefs_prior_mu=prior_mu,
        coefs_prior_cov=prior_cov,
        noise_precision=2.0,
    )
    model_batch = BayesianConjugateLinearRegressor(
        coefs_prior_mu=prior_mu,
        coefs_prior_cov=prior_cov,
        noise_precision=2.0,
    )

    model_sequential.fit(X1, y1)
    model_sequential.update(X2, y2)

    X_full = pd.concat([X1, X2], axis=0, ignore_index=True)
    y_full = pd.concat([y1, y2], axis=0, ignore_index=True)
    model_batch.fit(X_full, y_full)

    assert np.allclose(
        model_sequential._coefs_posterior_mu,
        model_batch._coefs_posterior_mu,
        atol=1e-10,
    )
    assert np.allclose(
        model_sequential._coefs_posterior_cov,
        model_batch._coefs_posterior_cov,
        atol=1e-10,
    )


@pytest.mark.skipif(
    not run_test_for_class(BayesianConjugateLinearRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_bayesian_conjugate_update_reduces_uncertainty():
    """Posterior covariance should shrink after assimilating new data."""
    rng = np.random.default_rng(123)
    n_features = 4

    X1 = pd.DataFrame(rng.normal(size=(30, n_features)))
    X2 = pd.DataFrame(rng.normal(size=(20, n_features)))

    true_w = np.array([[0.6], [-1.1], [0.2], [1.8]])
    y1 = pd.DataFrame(X1.values @ true_w + 0.5 * rng.normal(size=(30, 1)))
    y2 = pd.DataFrame(X2.values @ true_w + 0.5 * rng.normal(size=(20, 1)))

    model = BayesianConjugateLinearRegressor(
        coefs_prior_mu=np.zeros((n_features, 1)),
        coefs_prior_cov=np.eye(n_features),
        noise_precision=2.0,
    )

    model.fit(X1, y1)
    cov_before = model._coefs_posterior_cov.copy()

    model.update(X2, y2)
    cov_after = model._coefs_posterior_cov

    # uncertainty shrinks in trace and in PSD order
    assert np.trace(cov_after) <= np.trace(cov_before)

    eigvals = np.linalg.eigvalsh(cov_before - cov_after)
    assert np.all(eigvals >= -1e-12)
