"""Tests for model selection tuning utilities."""

import pandas as pd
import pytest
from sklearn.model_selection import KFold

from skpro.model_selection import GridSearchCV, RandomizedSearchCV
from skpro.regression.dummy import DummyProbaRegressor
from skpro.tests.test_switch import run_test_module_changed


def _get_test_data():
    X = pd.DataFrame({"x": [1, 2, 3, 4]})
    y = pd.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})
    return X, y


@pytest.mark.skipif(
    not run_test_module_changed("skpro.model_selection"),
    reason="Test only if skpro.model_selection has been changed",
)
def test_gridsearch_scoring_none_defaults_to_crps():
    """GridSearchCV should use CRPS when scoring is None."""
    X, y = _get_test_data()

    gscv = GridSearchCV(
        estimator=DummyProbaRegressor(),
        cv=KFold(n_splits=2),
        param_grid={"strategy": ["empirical"]},
        scoring=None,
    )
    gscv.fit(X, y)

    assert "mean_test_CRPS" in gscv.cv_results_.columns


def test_randomizedsearch_scoring_none_defaults_to_crps():
    """RandomizedSearchCV should use CRPS when scoring is None."""
    X, y = _get_test_data()

    rscv = RandomizedSearchCV(
        estimator=DummyProbaRegressor(),
        cv=KFold(n_splits=2),
        param_distributions={"strategy": ["empirical"]},
        n_iter=1,
        scoring=None,
        backend_params={"n_jobs": 1},
    )
    rscv.fit(X, y)

    assert "mean_test_CRPS" in rscv.cv_results_.columns


