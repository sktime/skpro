"""Tests for Nadaraya-Watson bandwidth handling."""

import numpy as np
import pandas as pd

from skpro.regression._bandwidth import bandwidth_1d
from skpro.regression.nonparametric import NadarayaWatsonCDE


def test_nadaraya_watson_target_bandwidth_isj_matches_helper():
    """ISJ target bandwidth should reuse the shared 1D helper."""
    X = pd.DataFrame({"x": np.linspace(-1.0, 1.0, 8)})
    y = pd.DataFrame({"y": [-1.0, -0.75, -0.1, 0.2, 0.9, 1.4, 1.8, 2.0]})

    reg = NadarayaWatsonCDE(y_kernel="gaussian", y_bandwidth="isj")
    reg.fit(X, y)

    expected = bandwidth_1d(y.values.ravel(), method="isj")

    assert np.isfinite(reg.y_bandwidth_)
    assert reg.y_bandwidth_ > 0
    assert np.isclose(reg.y_bandwidth_, expected)

    pred = reg.predict_proba(X.iloc[:2])
    assert np.isclose(pred._h, expected)


def test_nadaraya_watson_target_bandwidth_isj_degenerate_target_fallback():
    """Degenerate targets should still resolve to a positive smoothing bandwidth."""
    X = pd.DataFrame({"x": np.linspace(0.0, 1.0, 5)})
    y = pd.DataFrame({"y": np.ones(5)})

    reg = NadarayaWatsonCDE(y_kernel="gaussian", y_bandwidth="isj")
    reg.fit(X, y)

    assert np.isfinite(reg.y_bandwidth_)
    assert reg.y_bandwidth_ > 0
