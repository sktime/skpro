"""Tests for DummyProbaRegressor."""

import numpy as np
import pandas as pd
import pytest

from skpro.regression.dummy import DummyProbaRegressor, _resolve_kde_bw_method


def test_dummy_kernel_isj_fit_predict_proba_smoke():
    """Kernel strategy should support ISJ bandwidth in 1D."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(size=(80, 3)), columns=["a", "b", "c"])

    y1 = rng.normal(loc=-2.0, scale=0.4, size=40)
    y2 = rng.normal(loc=2.5, scale=0.6, size=40)
    y = pd.DataFrame(np.concatenate([y1, y2]), columns=["target"])

    reg = DummyProbaRegressor(strategy="kernel", bandwidth="isj", n_kde_samples=64)
    reg.fit(X, y)

    pred = reg.predict_proba(X.iloc[:5])
    assert pred is not None


def test_dummy_isj_warns_and_falls_back_for_multi_column_target():
    """ISJ should warn and fall back on multi-column targets."""
    y = np.arange(20, dtype=float).reshape(10, 2)

    with pytest.warns(UserWarning, match="only supported for 1D"):
        bw = _resolve_kde_bw_method(y, "isj")

    assert bw == "silverman"
