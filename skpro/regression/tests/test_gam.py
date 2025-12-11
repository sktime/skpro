"""Tests for GAM (Generalized Additive Model) regressor."""

import numpy as np
import pandas as pd
import pytest

from skpro.regression.gam import GAMRegressor
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(GAMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gam_normal_distribution():
    """Test GAM regressor with normal distribution."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:200]
    y = y.iloc[:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = GAMRegressor(distribution="normal", terms="auto")
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_proba = reg.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert y_pred_proba.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(GAMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gam_poisson_distribution():
    """Test GAM regressor with Poisson distribution."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    # Ensure positive values for Poisson
    y = y.abs() + 1
    X = X.iloc[:200]
    y = y.iloc[:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = GAMRegressor(distribution="poisson", link="log", terms="auto")
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_proba = reg.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert y_pred_proba.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(GAMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gam_gamma_distribution():
    """Test GAM regressor with Gamma distribution."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    # Ensure positive values for Gamma
    y = y.abs() + 1
    X = X.iloc[:200]
    y = y.iloc[:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = GAMRegressor(distribution="gamma", link="log", terms="auto")
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_proba = reg.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert y_pred_proba.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(GAMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gam_binomial_distribution():
    """Test GAM regressor with Binomial distribution."""
    import warnings

    from skpro.distributions.binomial import Binomial

    np.random.seed(42)

    # Generate synthetic data with proportions
    X = pd.DataFrame(
        {
            "x1": np.linspace(0, 10, 100),
            "x2": np.random.randn(100),
        }
    )

    # Generate binomial proportions (values in [0, 1])
    p_true = 1 / (1 + np.exp(-(-2 + 0.3 * X["x1"] + 0.2 * X["x2"])))
    y = pd.DataFrame(
        {"y": np.clip(p_true + np.random.normal(0, 0.05, 100), 0.01, 0.99)}
    )

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )

    # Suppress pygam's RuntimeWarning for McFadden R² calculation
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=RuntimeWarning, message=".*invalid value encountered.*"
        )

        reg = GAMRegressor(distribution="binomial", link="logit", terms="auto")
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        y_pred_proba = reg.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert y_pred_proba.shape == y_test.shape
    assert isinstance(y_pred_proba, Binomial)


@pytest.mark.skipif(
    not run_test_for_class(GAMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gam_with_custom_terms():
    """Test GAM regressor with custom spline terms."""
    from pygam import s
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:200]
    y = y.iloc[:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Custom terms: spline on first feature
    reg = GAMRegressor(distribution="normal", terms=s(0), max_iter=50)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_proba = reg.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert y_pred_proba.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(GAMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gam_predict_proba_returns_distribution():
    """Test that predict_proba returns proper distribution objects."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    from skpro.distributions.gamma import Gamma
    from skpro.distributions.normal import Normal
    from skpro.distributions.poisson import Poisson

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:100]
    y = y.iloc[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Test Normal distribution
    reg_normal = GAMRegressor(distribution="normal")
    reg_normal.fit(X_train, y_train)
    y_proba_normal = reg_normal.predict_proba(X_test)
    assert isinstance(y_proba_normal, Normal)

    # Test Poisson distribution
    y_pos = y.abs() + 1
    y_train_pos, y_test_pos = train_test_split(y_pos, random_state=0, test_size=0.25)
    X_train_pos = X_train
    X_test_pos = X_test

    reg_poisson = GAMRegressor(distribution="poisson", link="log")
    reg_poisson.fit(X_train_pos, y_train_pos)
    y_proba_poisson = reg_poisson.predict_proba(X_test_pos)
    assert isinstance(y_proba_poisson, Poisson)

    # Test Gamma distribution
    reg_gamma = GAMRegressor(distribution="gamma", link="log")
    reg_gamma.fit(X_train_pos, y_train_pos)
    y_proba_gamma = reg_gamma.predict_proba(X_test_pos)
    assert isinstance(y_proba_gamma, Gamma)
