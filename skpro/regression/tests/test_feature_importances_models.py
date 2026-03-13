"""Smoke tests for feature_importances on concrete regressors."""

import pandas as pd
import pytest

from skpro.tests.test_switch import run_test_for_class

from skpro.regression.ensemble import NGBoostRegressor
from skpro.regression.gam import GAMRegressor
from skpro.regression.cyclic_boosting import CyclicBoosting
from skpro.regression.xgboostlss import XGBoostLSS


def _small_diabetes(n=60):
    """Load a small diabetes subset as pandas DataFrames."""
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    return X.iloc[:n], pd.DataFrame(y).iloc[:n]


@pytest.mark.skipif(
    not run_test_for_class(NGBoostRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_ngboost_feature_importances_smoke():
    X, y = _small_diabetes(80)
    reg = NGBoostRegressor(n_estimators=15, learning_rate=0.1, verbose=False, verbose_eval=0)
    reg.fit(X, y)
    imp = reg.feature_importances()

    assert isinstance(imp, pd.Series)
    assert list(imp.index) == list(X.columns)
    assert len(imp) == X.shape[1]
    assert imp.name == "feature_importance"


@pytest.mark.skipif(
    not run_test_for_class(GAMRegressor),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_gam_feature_importances_smoke():
    X, y = _small_diabetes(80)
    reg = GAMRegressor(distribution="normal", terms="auto", max_iter=80)
    reg.fit(X, y)
    imp = reg.feature_importances()

    assert isinstance(imp, pd.Series)
    assert list(imp.index) == list(X.columns)
    assert len(imp) == X.shape[1]
    assert imp.name == "feature_importance"


@pytest.mark.skipif(
    not run_test_for_class(CyclicBoosting),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cyclic_boosting_feature_importances_smoke():
    X, y = _small_diabetes(60)
    reg = CyclicBoosting(maximal_iterations=3, alpha=0.25)
    reg.fit(X, y)
    imp = reg.feature_importances()

    assert isinstance(imp, pd.Series)
    assert list(imp.index) == list(X.columns)
    assert len(imp) == X.shape[1]
    assert imp.name == "feature_importance"


@pytest.mark.skipif(
    not run_test_for_class(XGBoostLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_xgboostlss_feature_importances_smoke():
    X, y = _small_diabetes(50)
    reg = XGBoostLSS(
        n_trials=0,
        num_boost_round=10,
        eta=0.1,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    reg.fit(X, y)
    imp = reg.feature_importances()

    assert isinstance(imp, pd.Series)
    assert list(imp.index) == list(X.columns)
    assert len(imp) == X.shape[1]
    assert imp.name == "feature_importance"
