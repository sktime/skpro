"""Tests for cyclic boosting regressor."""

import pytest

from skpro.regression.cyclic_boosting import CyclicBoosting
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(CyclicBoosting),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_cyclic_boosting_simple_use():
    """Test simple use of cyclic boosting regressor."""
    from cyclic_boosting import flags
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    from skpro.regression.cyclic_boosting import CyclicBoosting

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)

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

    reg_proba = CyclicBoosting(feature_properties=fp)
    reg_proba.fit(X_train, y_train)
    y_pred = reg_proba.predict_proba(X_test)

    assert y_pred.shape == y_test.shape