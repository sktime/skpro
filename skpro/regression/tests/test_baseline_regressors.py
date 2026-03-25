import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from skpro.regression.deterministic_reduction import DeterministicReductionRegressor
from skpro.regression.unconditional_distfit import UnconditionalDistfitRegressor


@pytest.mark.skipif(
    not pytest.importorskip("distfit", reason="distfit required"),
    reason="distfit not installed",
)
def test_unconditional_distfit_regressor():
    X = np.random.randn(100, 3)
    y = np.random.randn(100)
    reg = UnconditionalDistfitRegressor(distr_type="norm")
    reg.fit(X, y)
    dist = reg.predict_proba(X)
    samples = dist.sample(10)
    assert samples.shape[0] == 10
    assert hasattr(dist, "pdf")
    assert hasattr(dist, "mean")
    assert hasattr(dist, "var")


def test_deterministic_reduction_regressor_gaussian():
    X = np.random.randn(100, 2)
    y = np.random.randn(100)
    reg = DeterministicReductionRegressor(LinearRegression(), distr_type="gaussian")
    reg.fit(X, y)
    dist = reg.predict_proba(X)
    assert hasattr(dist, "mean")
    assert hasattr(dist, "sigma")
    assert np.allclose(dist.sigma, np.sqrt(np.var(y)))


def test_deterministic_reduction_regressor_laplace():
    X = np.random.randn(100, 2)
    y = np.random.randn(100)
    reg = DeterministicReductionRegressor(LinearRegression(), distr_type="laplace")
    reg.fit(X, y)
    dist = reg.predict_proba(X)
    assert hasattr(dist, "mu")
    assert hasattr(dist, "scale")
    assert np.allclose(dist.scale, np.sqrt(np.var(y) / 2))


@pytest.mark.xfail(
    reason="distfit KDE support is broken with recent scipy (scipy.stats.kde removed).",
    strict=False,
)
def test_unconditional_distfit_regressor_kde():
    # Test KDE as a nonparametric option if distfit supports it
    # Broken in distfit due to scipy.stats.kde removal in recent scipy versions.
    X = np.random.randn(50, 2)
    y = np.random.randn(50)
    reg = UnconditionalDistfitRegressor(distr_type="kde")
    reg.fit(X, y)
    dist = reg.predict_proba(X)
    samples = dist.sample(5)
    assert samples.shape[0] == 5


# Note: duplicate test removed to fix F811
