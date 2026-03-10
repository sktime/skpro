"""Tests for the LightGBMLSS regressor."""

import pandas as pd
import pytest

from skpro.regression.lightgbmlss import LightGBMLSS
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(LightGBMLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_lightgbmlss_params_no_optuna():
    """Test simple use of LightGBMLSS regressor."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:200]
    y = y.iloc[:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg_proba = LightGBMLSS(n_trials=0, num_boost_round=5, num_leaves=7, n_jobs=1)
    reg_proba.fit(X_train, y_train)

    y_pred = reg_proba.predict_proba(X_test)

    assert reg_proba.lgb_params_["num_leaves"] == 7
    assert reg_proba.lgb_params_["num_threads"] == 1
    assert reg_proba.lgblss_.booster.current_iteration() == 5
    assert y_pred.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(LightGBMLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "params, expected_lgb_params, expected_rounds, should_error",
    [
        ({"n_jobs": 1}, {"num_threads": 1}, 100, False),
        ({"num_threads": 2}, {"num_threads": 2}, 100, False),
        ({"n_estimators": 7}, {}, 7, False),
        ({"n_jobs": 1, "num_threads": 2}, {}, None, True),
        ({"n_estimators": 7, "num_boost_round": 10}, {}, None, True),
    ],
)
def test_lightgbmlss_param_handling(
    params, expected_lgb_params, expected_rounds, should_error
):
    """Test LightGBM parameter aliases and training params."""
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:50]
    y = y.iloc[:50]

    reg = LightGBMLSS(n_trials=0, **params)

    if should_error:
        with pytest.raises(ValueError):
            reg.fit(X, y)
    else:
        reg.fit(X, y)
        y_pred = reg.predict_proba(X)

        assert y_pred.shape == y.shape
        assert reg.num_boost_round_ == expected_rounds

        for key, value in expected_lgb_params.items():
            assert reg.lgb_params_.get(key) == value


@pytest.mark.skipif(
    not run_test_for_class(LightGBMLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_lightgbmlss_nonconsecutive_index():
    """Test fit works with non-consecutive pandas indices."""
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)

    X = X.iloc[:120].iloc[::2]
    y = y.iloc[:120].iloc[::2]

    reg = LightGBMLSS(n_trials=0, num_boost_round=5, num_leaves=7, n_jobs=1)
    reg.fit(X, y)
    y_pred = reg.predict_proba(X.iloc[:10])

    assert list(X.index[:5]) != list(range(5))
    assert y_pred.shape == (10, 1)


@pytest.mark.skipif(
    not run_test_for_class(LightGBMLSS),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_lightgbmlss_hyperopt_smoke():
    """Test the hyperparameter optimization path."""
    from sklearn.datasets import load_diabetes

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:80]
    y = y.iloc[:80]

    reg = LightGBMLSS(
        n_trials=1,
        max_minutes=1,
        num_boost_round=5,
        n_jobs=1,
    )
    reg.fit(X, y)
    y_pred = reg.predict_proba(X.iloc[:10])

    assert y_pred.shape == (10, 1)
    assert reg.lgb_params_["num_threads"] == 1
    assert reg.lgblss_.booster.current_iteration() >= 1
