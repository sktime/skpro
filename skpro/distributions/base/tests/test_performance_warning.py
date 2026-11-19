
import pytest
import pandas as pd
import numpy as np
import warnings
from skpro.distributions.base._base import _get_fast_index_np


def test_warn_on_tz_aware():
    """Test that UserWarning is raised for tz-aware DatetimeIndex."""
    dt_idx = pd.date_range("2020-01-01", periods=5, tz="UTC")
    with pytest.warns(UserWarning, match="known to be slow"):
        _get_fast_index_np(dt_idx)


def test_warn_on_period():
    """Test that UserWarning is raised for PeriodIndex."""
    period_idx = pd.period_range("2020-01-01", periods=5, freq="D")
    with pytest.warns(UserWarning, match="known to be slow"):
        _get_fast_index_np(period_idx)


def test_warn_on_interval():
    """Test that UserWarning is raised for IntervalIndex."""
    interval_idx = pd.interval_range(start=0, end=5)
    with pytest.warns(UserWarning, match="known to be slow"):
        _get_fast_index_np(interval_idx)


def test_no_warn_on_standard():
    """Test that no warning is raised for standard Index types."""
    # Standard Int64 Index
    std_idx = pd.Index([1, 2, 3, 4, 5])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _get_fast_index_np(std_idx)
        relevant_warnings = [
            x for x in w if "known to be slow" in str(x.message)
        ]
        msg = f"Unexpected warnings: {[str(x.message) for x in relevant_warnings]}"
        assert len(relevant_warnings) == 0, msg

    # RangeIndex
    range_idx = pd.RangeIndex(start=0, stop=5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _get_fast_index_np(range_idx)
        relevant_warnings = [
            x for x in w if "known to be slow" in str(x.message)
        ]
        assert len(relevant_warnings) == 0

    # Naive DatetimeIndex
    naive_dt_idx = pd.date_range("2020-01-01", periods=5)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _get_fast_index_np(naive_dt_idx)
        relevant_warnings = [
            x for x in w if "known to be slow" in str(x.message)
        ]
        assert len(relevant_warnings) == 0


def test_casting_correctness():
    """Test that the casting logic actually returns correct numpy arrays."""
    dt_idx = pd.date_range("2020-01-01", periods=3, tz="UTC")
    with pytest.warns(UserWarning):
        res = _get_fast_index_np(dt_idx)

    assert isinstance(res, np.ndarray)
    assert np.issubdtype(res.dtype, np.datetime64)
    expected = dt_idx.tz_localize(None).to_numpy()
    np.testing.assert_array_equal(res, expected)

    # PeriodIndex -> should become object array of Periods
    period_idx = pd.period_range("2020-01-01", periods=3, freq="D")
    with pytest.warns(UserWarning):
        res = _get_fast_index_np(period_idx)

    assert isinstance(res, np.ndarray)
    assert res.dtype == object
    assert isinstance(res[0], pd.Period)

    # IntervalIndex -> should become object array of Intervals
    interval_idx = pd.interval_range(start=0, end=3)
    with pytest.warns(UserWarning):
        res = _get_fast_index_np(interval_idx)

    assert isinstance(res, np.ndarray)
    assert res.dtype == object
    assert isinstance(res[0], pd.Interval)
