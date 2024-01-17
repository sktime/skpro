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

    from skpro.regression.cyclic_boosting import CyclicBoosting

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    y = pd.DataFrame(y)
    X = X.iloc[:50]
    y = y.iloc[:50]
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

    reg_proba = CyclicBoosting(feature_groups=features)
    reg_proba.fit(X_train, y_train)
    y_pred = reg_proba.predict_proba(X_test)

    assert y_pred.shape == y_test.shape
