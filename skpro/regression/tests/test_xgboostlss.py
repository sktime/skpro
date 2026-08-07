"""Tests for the XGBoostLSS regressor."""

import pandas as pd
import pytest

from skpro.regression.xgboostlss import XGBoostLSS
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(XGBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_xgboostlss_params_no_optuna():
    """Test simple use of XGBoostLSS regressor."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:200]
    y = y.iloc[:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg_proba = XGBoostLSS(n_trials=0, max_depth=2)
    reg_proba.fit(X_train, y_train)

    y_pred = reg_proba.predict_proba(X_test)
    trees_df = reg_proba.xgblss_.booster.trees_to_dataframe()
    max_nodes_per_tree = trees_df.groupby("Tree")["Node"].max()
    # All trees should have Node max <= 6 (for max_depth=2)
    assert (max_nodes_per_tree <= 6).all()
    assert y_pred.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(XGBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "params, expected_xgb_params, should_error",
    [
        ({"learning_rate": 0.1}, {"eta": 0.1}, False),
        ({"eta": 0.2}, {"eta": 0.2}, False),
        ({"n_estimators": 50}, {}, False),  # not an xgb_param
        ({"learning_rate": 0.1, "eta": 0.2}, {}, True),
        ({"n_estimators": 50, "num_boost_round": 200}, {}, True),
    ],
)
def test_xgboostlss_param_handling(params, expected_xgb_params, should_error):
    """Test parameter aliases and training params."""
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:50]  # smaller for speed
    y = y.iloc[:50]

    reg = XGBoostLSS(n_trials=0, **params)

    if should_error:
        with pytest.raises(ValueError):
            reg.fit(X, y)
    else:
        reg.fit(X, y)
        y_pred = reg.predict_proba(X)
        assert y_pred.shape == y.shape

        for key, value in expected_xgb_params.items():
            assert reg.xgb_params_.get(key) == value
