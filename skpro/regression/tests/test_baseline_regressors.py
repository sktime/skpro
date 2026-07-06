import importlib.util

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from skpro.regression.deterministic_reduction import DeterministicReductionRegressor
from skpro.regression.unconditional_distfit import UnconditionalDistfitRegressor

HAS_DISTFIT = importlib.util.find_spec("distfit") is not None

requires_distfit = pytest.mark.skipif(not HAS_DISTFIT, reason="distfit required")


@requires_distfit
def test_unconditional_distfit_regressor_invalid_distr_type():
    with pytest.raises(ValueError, match="distr_type"):
        UnconditionalDistfitRegressor(distr_type="not_a_dist")


@requires_distfit
def test_unconditional_distfit_regressor_fit_and_predict():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, 3))
    y = rng.normal(size=100)

    reg = UnconditionalDistfitRegressor(distr_type="norm", random_state=42)
    reg.fit(X, y)
    dist = reg.predict_proba(X)

    samples = dist.sample(10)
    assert samples.shape[0] == 10 * len(X)
    assert hasattr(dist, "pdf")
    assert hasattr(dist, "mean")
    assert hasattr(dist, "var")


@requires_distfit
def test_unconditional_distfit_distribution_parameters_and_mean():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(120, 2))
    y = rng.normal(loc=2.5, scale=1.3, size=120)

    reg = UnconditionalDistfitRegressor(distr_type="laplace", random_state=123)
    reg.fit(X, y)
    dist = reg.predict_proba(X)

    model = dist.distfit_obj.model
    assert isinstance(model, dict)
    assert "loc" in model
    assert "scale" in model
    # distfit is fit on y only.
    # mean of returned distribution should match fitted location.
    assert np.allclose(dist.mean().values, model["loc"])


@requires_distfit
def test_unconditional_distfit_regressor_multioutput_raises():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 3))
    y = rng.normal(size=(100, 2))

    reg = UnconditionalDistfitRegressor(distr_type="norm")
    with pytest.raises(NotImplementedError, match="univariate"):
        reg.fit(X, y)


@requires_distfit
def test_unconditional_distfit_regressor_supports_numpy_arrays():
    rng = np.random.default_rng(99)
    X = rng.normal(size=(50, 2))
    y = rng.normal(size=50)

    reg = UnconditionalDistfitRegressor(distr_type="norm")
    reg.fit(X, y)
    dist = reg.predict_proba(X)

    assert hasattr(dist, "mean")


def test_deterministic_reduction_regressor_gaussian():
    rng = np.random.default_rng(24)
    X = rng.normal(size=(100, 2))
    y = rng.normal(size=100)

    reg = DeterministicReductionRegressor(LinearRegression(), distr_type="gaussian")
    reg.fit(X, y)
    dist = reg.predict_proba(X)

    assert hasattr(dist, "mean")
    assert hasattr(dist, "sigma")
    assert np.allclose(dist.sigma, np.sqrt(np.var(y)))


def test_deterministic_reduction_regressor_invalid_distr_type():
    with pytest.raises(ValueError, match="distr_type"):
        DeterministicReductionRegressor(LinearRegression(), distr_type="not_a_dist")


def test_deterministic_reduction_regressor_multioutput_raises():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(100, 2))
    y = rng.normal(size=(100, 2))

    reg = DeterministicReductionRegressor(LinearRegression(), distr_type="gaussian")
    with pytest.raises(NotImplementedError, match="univariate"):
        reg.fit(X, y)


def test_deterministic_reduction_regressor_supports_numpy_arrays():
    rng = np.random.default_rng(101)
    X = rng.normal(size=(50, 2))
    y = rng.normal(size=50)

    reg = DeterministicReductionRegressor(LinearRegression(), distr_type="gaussian")
    reg.fit(X, y)
    dist = reg.predict_proba(X)

    assert hasattr(dist, "mean")


@pytest.mark.parametrize("distr_type", ["gaussian", "laplace"])
def test_deterministic_reduction_distribution_correctness(distr_type):
    rng = np.random.default_rng(202)
    X = rng.normal(size=(120, 3))
    y = 1.5 * X[:, 0] - 0.3 * X[:, 1] + rng.normal(scale=0.2, size=120)

    reg = DeterministicReductionRegressor(LinearRegression(), distr_type=distr_type)
    reg.fit(X, y)
    dist = reg.predict_proba(X)

    pred = reg.regressor_.predict(X).reshape(-1, 1)
    mean_df = dist.mean()

    assert np.allclose(mean_df.values, pred)
    if distr_type == "gaussian":
        assert hasattr(dist, "sigma")
        assert np.allclose(dist.sigma, np.sqrt(np.var(y)))
    else:
        assert hasattr(dist, "scale")
        assert np.allclose(dist.scale, np.sqrt(np.var(y) / 2))


def test_deterministic_reduction_regressor_laplace():
    rng = np.random.default_rng(77)
    X = rng.normal(size=(100, 2))
    y = rng.normal(size=100)

    reg = DeterministicReductionRegressor(LinearRegression(), distr_type="laplace")
    reg.fit(X, y)
    dist = reg.predict_proba(X)

    assert hasattr(dist, "mu")
    assert hasattr(dist, "scale")
    assert np.allclose(dist.scale, np.sqrt(np.var(y) / 2))
