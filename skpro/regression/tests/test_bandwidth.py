"""Tests for bandwidth and noise-schedule utilities."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import time

import numpy as np
import pytest

from skpro.regression._bandwidth import (
    bandwidth_1d,
    bw_isj_1d,
    bw_scott_1d,
    bw_silverman_1d,
)
from skpro.tests.test_switch import run_test_module_changed

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("skpro.regression"),
    reason="run only if skpro.regression has been changed",
)


def test_bandwidth_1d_methods_return_finite_positive_values():
    """Test all methods return finite positive bandwidths on non-degenerate data."""
    rng = np.random.default_rng(42)
    y = np.r_[rng.normal(-1.0, 0.7, size=250), rng.normal(2.0, 0.4, size=250)]

    for method in ["silverman", "scott", "isj"]:
        h = bandwidth_1d(y, method=method)
        assert np.isfinite(h)
        assert h > 0


def test_bandwidth_weight_handling():
    """Test weighted inputs for Scott and Silverman."""
    rng = np.random.default_rng(0)
    y = rng.normal(0.0, 1.0, size=200)
    weights = np.abs(rng.normal(0.0, 1.0, size=200))
    weights[:20] = 0.0

    h_scott = bandwidth_1d(y, method="scott", weights=weights)
    h_silverman = bandwidth_1d(y, method="silverman", weights=weights)

    assert h_scott > 0
    assert h_silverman > 0


def test_bw_helpers_match_dispatcher():
    """Test helper functions match generic dispatcher outputs."""
    rng = np.random.default_rng(7)
    y = rng.normal(size=300)

    assert np.isclose(bw_scott_1d(y), bandwidth_1d(y, method="scott"))
    assert np.isclose(bw_silverman_1d(y), bandwidth_1d(y, method="silverman"))
    assert np.isclose(bw_isj_1d(y), bandwidth_1d(y, method="isj"))


def test_bandwidth_runtime_smoke_comparison():
    """Performance smoke test for selector runtime and ISJ comparison."""
    rng = np.random.default_rng(123)
    y = np.r_[rng.normal(-2.0, 0.8, size=1000), rng.normal(1.0, 1.2, size=1000)]

    methods = ["scott", "silverman", "isj"]
    times = {}
    values = {}
    for method in methods:
        start = time.perf_counter()
        values[method] = bandwidth_1d(y, method=method)
        times[method] = time.perf_counter() - start

    for method in methods:
        assert np.isfinite(values[method])
        assert values[method] > 0

    # Broad bounds to catch pathological slowdowns while avoiding flaky failures.
    assert times["scott"] < 1.0
    assert times["silverman"] < 1.0
    assert times["isj"] < 5.0
    assert times["isj"] > min(times["scott"], times["silverman"])


def test_legacy_aliases_rejected():
    """Legacy aliases should be rejected to enforce canonical naming."""
    with pytest.raises(ValueError, match="Unknown method"):
        bandwidth_1d(np.array([0.0, 1.0, 2.0]), method="sqrt_decay")
