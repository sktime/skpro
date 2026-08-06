"""Tests for BayesianRidgeRegressor and BaseBayesianRegressor API."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from skpro.distributions.normal import Normal
from skpro.regression.bayesian._base import BaseBayesianRegressor
from skpro.regression.bayesian._prior import Prior
from skpro.regression.bayesian._ridge import BayesianRidgeRegressor


@pytest.fixture
def regression_data():
    """Generate simple linear regression data with known coefficients."""
    rng = np.random.default_rng(42)
    n, d = 100, 3
    X = rng.standard_normal((n, d))
    true_coefs = np.array([1.5, -2.0, 0.5])
    y = X @ true_coefs + 0.1 * rng.standard_normal(n)

    X_df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    y_df = pd.DataFrame(y, columns=["target"])
    return X_df, y_df, true_coefs


class TestBayesianRidgeRegressor:
    """Tests for BayesianRidgeRegressor."""

    def test_fit_predict(self, regression_data):
        """Test basic fit and predict_proba returns a Normal distribution."""
        X, y, _ = regression_data
        reg = BayesianRidgeRegressor(n_iter=100)
        reg.fit(X, y)

        dist = reg.predict_proba(X)
        assert isinstance(dist, Normal)

        # Point predictions should be a DataFrame with correct shape
        y_pred = reg.predict(X)
        assert isinstance(y_pred, pd.DataFrame)
        assert y_pred.shape == y.shape

    def test_posterior_recovers_true_coefs(self, regression_data):
        """Test that posterior mean is close to true coefficients."""
        X, y, true_coefs = regression_data
        reg = BayesianRidgeRegressor(n_iter=300)
        reg.fit(X, y)

        np.testing.assert_allclose(reg.coef_mean_, true_coefs, atol=0.2)

    def test_get_posterior(self, regression_data):
        """Test get_posterior returns dict of distributions."""
        X, y, _ = regression_data
        reg = BayesianRidgeRegressor()
        reg.fit(X, y)

        posterior = reg.get_posterior()
        assert isinstance(posterior, dict)
        assert "coefficients" in posterior
        assert isinstance(posterior["coefficients"], Normal)

    def test_get_posterior_summary(self, regression_data):
        """Test posterior summary returns DataFrame with expected columns."""
        X, y, _ = regression_data
        reg = BayesianRidgeRegressor()
        reg.fit(X, y)

        summary = reg.get_posterior_summary()
        assert isinstance(summary, pd.DataFrame)
        assert "mean" in summary.columns
        assert "std" in summary.columns
        assert "q_0.025" in summary.columns
        assert "q_0.975" in summary.columns

    def test_sample_posterior(self, regression_data):
        """Test sample_posterior returns parameter samples."""
        X, y, _ = regression_data
        reg = BayesianRidgeRegressor()
        reg.fit(X, y)

        samples = reg.sample_posterior(n_samples=50)
        assert isinstance(samples, dict)
        assert "coefficients" in samples

    def test_sequential_update(self, regression_data):
        """Test that update changes the posterior."""
        X, y, _ = regression_data
        X1, X2 = X.iloc[:50], X.iloc[50:]
        y1, y2 = y.iloc[:50], y.iloc[50:]

        reg = BayesianRidgeRegressor(n_iter=100)
        reg.fit(X1, y1)
        coef_before = reg.coef_mean_.copy()

        reg.update(X2, y2)
        coef_after = reg.coef_mean_.copy()

        # Posterior should change after seeing more data
        assert not np.allclose(coef_before, coef_after)

    def test_get_test_params(self):
        """Test that get_test_params returns valid parameter sets."""
        params_list = BayesianRidgeRegressor.get_test_params()
        assert isinstance(params_list, list)
        assert len(params_list) >= 1

        for params in params_list:
            reg = BayesianRidgeRegressor(**params)
            assert isinstance(reg, BaseBayesianRegressor)

    def test_inherits_base_bayesian(self):
        """Test that BayesianRidgeRegressor is a BaseBayesianRegressor."""
        reg = BayesianRidgeRegressor()
        assert isinstance(reg, BaseBayesianRegressor)


class TestPrior:
    """Tests for the Prior specification class."""

    def test_prior_wraps_distribution(self):
        """Test Prior wraps a skpro distribution."""
        dist = Normal(mu=0, sigma=1)
        prior = Prior(dist, name="weights")
        assert prior.name == "weights"
        assert prior.distribution is dist

    def test_prior_rejects_non_distribution(self):
        """Test Prior raises TypeError for non-distribution inputs."""
        with pytest.raises(TypeError, match="BaseDistribution"):
            Prior(42)

    def test_prior_repr(self):
        """Test Prior has informative repr."""
        prior = Prior(Normal(mu=0, sigma=1), name="w")
        assert "Normal" in repr(prior)
        assert "w" in repr(prior)
