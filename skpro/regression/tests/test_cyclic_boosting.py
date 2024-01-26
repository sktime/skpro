"""Tests for cyclic boosting regressor."""

import pandas as pd
import pytest

from skpro.regression.cyclic_boosting import CyclicBoosting
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(CyclicBoosting),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cyclic_boosting_simple_use():
    """Test simple use of cyclic boosting regressor."""
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:200]
    y = y.iloc[:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    reg_proba = CyclicBoosting()
    reg_proba.fit(X_train, y_train)
    y_pred = reg_proba.predict_proba(X_test)

    assert y_pred.shape == y_test.shape


@pytest.mark.skipif(
    not run_test_for_class(CyclicBoosting),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cyclic_boosting_with_manual_paramaters():
    """Test use of cyclic boosting regressor with_manual_paramaters."""
    from cyclic_boosting import flags
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:200]
    y = y.iloc[:200]
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    features = [
        "age",
        "sex",
        "bmi",
        "bp",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        ("age", "sex"),
    ]

    fp = {
        "age": flags.IS_CONTINUOUS,
        "sex": flags.IS_CONTINUOUS,
        "bmi": flags.IS_CONTINUOUS,
        "bp": flags.IS_CONTINUOUS,
        "s1": flags.IS_CONTINUOUS,
        "s2": flags.IS_CONTINUOUS,
        "s3": flags.IS_CONTINUOUS,
        "s4": flags.IS_CONTINUOUS,
        "s5": flags.IS_CONTINUOUS,
        "s6": flags.IS_CONTINUOUS,
    }

    reg_proba = CyclicBoosting(
        feature_groups=features,
        feature_properties=fp,
        maximal_iterations=5,
        alpha=0.25,
        mode="additive",
        bound="S",
        lower=0.0,
    )
    reg_proba.fit(X_train, y_train)
    y_pred = reg_proba.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
