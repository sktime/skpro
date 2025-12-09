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
