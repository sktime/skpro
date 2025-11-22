"""Tests Generalized Additive Model regressor."""

import pandas as pd
import pytest

from skpro.regression.gam import GAMRegressor
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(GAMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gam_simple_use():
    """Test simple use of GAM regressor with normal distribution."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:100]
    y = y.iloc[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    gam_reg = GAMRegressor(distribution="normal", link="identity")
    gam_reg.fit(X_train, y_train)
    y_pred = gam_reg.predict(X_test)
    y_pred_proba = gam_reg.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
    assert y_pred_proba.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(GAMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gam_distributions():
    """Test GAM regressor with different distributions."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:100]
    y = y.iloc[:100]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # test normal distribution
    gam_normal = GAMRegressor(distribution="normal", link="identity")
    gam_normal.fit(X_train, y_train)
    assert hasattr(gam_normal, "_actual_distribution")
    assert gam_normal._actual_distribution == "normal"
    y_pred_normal = gam_normal.predict(X_test)
    y_pred_proba_normal = gam_normal.predict_proba(X_test)
    assert y_pred_normal.shape == y_test.shape
    assert y_pred_proba_normal.shape == y_test.shape

    # test poisson distribution
    gam_poisson = GAMRegressor(distribution="poisson", link="log")
    gam_poisson.fit(X_train, y_train)
    assert gam_poisson._actual_distribution == "poisson"
    y_pred_poisson = gam_poisson.predict(X_test)
    y_pred_proba_poisson = gam_poisson.predict_proba(X_test)
    assert y_pred_poisson.shape == y_test.shape
    assert y_pred_proba_poisson.shape == y_test.shape

    # test gamma distribution
    gam_gamma = GAMRegressor(distribution="gamma", link="log")
    gam_gamma.fit(X_train, y_train)
    assert gam_gamma._actual_distribution == "gamma"
    y_pred_gamma = gam_gamma.predict(X_test)
    y_pred_proba_gamma = gam_gamma.predict_proba(X_test)
    assert y_pred_gamma.shape == y_test.shape
    assert y_pred_proba_gamma.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(GAMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gam_get_test_params():
    """Test that GAMRegressor.get_test_params works correctly."""
    params = GAMRegressor.get_test_params()
    assert params is not None

    # if pygam isn't installed, it returns a special marker
    if isinstance(params, dict) and params.get("distribution") == "runtests-no-pygam":
        return

    # otherwise should get a list of parameter sets
    assert isinstance(params, list)
    assert len(params) > 0

    # each param set should create a valid instance
    for param_set in params:
        gam = GAMRegressor(**param_set)
        assert isinstance(gam, GAMRegressor)
