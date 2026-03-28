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
    def test_unconditional_distfit_regressor_invalid_distr_type():
        with pytest.raises(ValueError):
            UnconditionalDistfitRegressor(distr_type="not_a_dist")

    def test_unconditional_distfit_regressor_multioutput():
        X = np.random.randn(100, 3)
        y = np.random.randn(100, 2)
        reg = UnconditionalDistfitRegressor(distr_type="norm")
        with pytest.raises(NotImplementedError):
            reg.fit(X, y)

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
    def test_deterministic_reduction_regressor_invalid_distr_type():
        with pytest.raises(ValueError):
            DeterministicReductionRegressor(LinearRegression(), distr_type="not_a_dist")

    def test_deterministic_reduction_regressor_multioutput():
        X = np.random.randn(100, 2)
        y = np.random.randn(100, 2)
        reg = DeterministicReductionRegressor(LinearRegression(), distr_type="gaussian")
        with pytest.raises(NotImplementedError):
            reg.fit(X, y)

    X = np.random.randn(100, 2)
    y = np.random.randn(100)
    reg = DeterministicReductionRegressor(LinearRegression(), distr_type="gaussian")
    reg.fit(X, y)
    dist = reg.predict_proba(X)
    assert hasattr(dist, "mean")
    assert hasattr(dist, "sigma")
    assert np.allclose(dist.sigma, np.sqrt(np.var(y)))


def test_deterministic_reduction_regressor_laplace():
    def test_unconditional_distfit_regressor_non_dataframe():
        # Should work with numpy arrays as y
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        reg = UnconditionalDistfitRegressor(distr_type="norm")
        reg.fit(X, y)
        dist = reg.predict_proba(X)
        assert hasattr(dist, "mean")

    def test_deterministic_reduction_regressor_non_dataframe():
        # Should work with numpy arrays as X and y
        X = np.random.randn(50, 2)
        y = np.random.randn(50)
        reg = DeterministicReductionRegressor(LinearRegression(), distr_type="gaussian")
        reg.fit(X, y)
        dist = reg.predict_proba(X)
        assert hasattr(dist, "mean")

    X = np.random.randn(100, 2)
    y = np.random.randn(100)
    reg = DeterministicReductionRegressor(LinearRegression(), distr_type="laplace")
    reg.fit(X, y)
    dist = reg.predict_proba(X)
    assert hasattr(dist, "mu")
    assert hasattr(dist, "scale")
    assert np.allclose(dist.scale, np.sqrt(np.var(y) / 2))
