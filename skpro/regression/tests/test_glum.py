"""Tests Glum regressor."""

import pytest

from skpro.regression.linear import GlumRegressor
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(GlumRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_glum_simple_use():
    """Test simple use of Glum regressor."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.iloc[:200]
    y = y.iloc[:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = GlumRegressor(family="normal")
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_pred_proba = reg.predict_proba(X_test)

    assert len(y_pred) == len(y_test)
    assert len(y_pred_proba) == len(y_test)
