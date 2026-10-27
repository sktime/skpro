"""Regression tests for Nadaraya-Watson prediction index alignment."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from skpro.regression.nonparametric import NadarayaWatsonCDE
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(NadarayaWatsonCDE),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize(
    "index",
    [
        pd.Index(["second", "first"], name="case"),
        pd.date_range("2025-01-01", periods=2, tz="Asia/Kolkata", name="time"),
        pd.MultiIndex.from_tuples([("west", 2), ("east", 1)], names=["site", "case"]),
        pd.MultiIndex.from_arrays(
            [
                ["west", "east"],
                pd.date_range("2025-01-01", periods=2, tz="Asia/Kolkata"),
                [2, 1],
            ],
            names=["site", "time", "case"],
        ),
    ],
    ids=["string", "datetime", "multiindex", "multiindex-datetime"],
)
def test_empirical_predictions_preserve_index(index):
    """Empirical predictions preserve row keys and the conditional moments."""
    X_train = pd.DataFrame({"x": [-1.0, 0.0, 1.0, 2.0]})
    y_train = pd.DataFrame({"target": [0.0, 2.0, 5.0, 9.0]})
    X_test = pd.DataFrame({"x": [1.5, -0.5]}, index=index)
    reg = NadarayaWatsonCDE(bandwidth=1.0, normalize_features=False)
    reg.fit(X_train, y_train)

    proba = reg.predict_proba(X_test)

    # Independently compute the Gaussian kernel weights and both moments.
    distances = X_test.to_numpy() - X_train.to_numpy().T
    weights = np.exp(-0.5 * distances**2)
    weights /= weights.sum(axis=1, keepdims=True)
    values = y_train.to_numpy()
    mean = weights @ values
    variance = weights @ (values**2) - mean**2
    expected_mean = pd.DataFrame(mean, index=index, columns=y_train.columns)
    expected_var = pd.DataFrame(variance, index=index, columns=y_train.columns)

    pd.testing.assert_index_equal(proba.index, index)
    pd.testing.assert_frame_equal(proba.mean(), expected_mean, check_freq=False)
    pd.testing.assert_frame_equal(proba.mean(), reg.predict(X_test), check_freq=False)
    pd.testing.assert_frame_equal(proba.var(), expected_var, check_freq=False)
    pd.testing.assert_frame_equal(
        reg.predict_var(X_test), expected_var, check_freq=False
    )
    quantiles = reg.predict_quantiles(X_test, alpha=[0.25, 0.75])
    pd.testing.assert_index_equal(quantiles.index, index)
    np.testing.assert_array_equal(quantiles.iloc[:, 0], [5.0, 0.0])
    np.testing.assert_array_equal(quantiles.iloc[:, 1], [9.0, 2.0])
