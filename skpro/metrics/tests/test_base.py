"""Tests for the base class of probabilistic metrics."""

import numpy as np
import pandas as pd
import pytest

from skpro.distributions import Normal
from skpro.metrics import CRPS, IntervalWidth
from skpro.metrics.base import BaseDistrMetric, BaseProbaMetric


class _MeanIntervalWidth(BaseProbaMetric):
    """Interval metric implementing only _evaluate, for testing the fallback.

    Mean width of the predicted intervals, averaged over rows.
    """

    _tags = {"scitype:y_pred": "pred_interval", "lower_is_better": True}

    def _evaluate(self, y_true, y_pred, **kwargs):
        lower = y_pred.xs("lower", axis=1, level=2)
        upper = y_pred.xs("upper", axis=1, level=2)
        return pd.DataFrame((upper - lower).mean(axis=0)).T


class _MeanCRPS(BaseDistrMetric):
    """Distribution metric implementing only _evaluate, for testing the fallback.

    CRPS averaged over rows, computed via the energy of the distribution.
    """

    def __init__(self, multioutput="uniform_average", multivariate=False):
        self.multivariate = multivariate
        super().__init__(multioutput=multioutput)

    def _evaluate(self, y_true, y_pred, **kwargs):
        y_true_np = y_true.to_numpy()
        crps = y_pred.energy(y_true_np) - 0.5 * y_pred.energy()
        return pd.DataFrame(crps.mean(axis=0)).T


def _make_interval_data():
    y_true = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=[10, 11, 12, 13])
    cols = pd.MultiIndex.from_tuples(
        [
            ("a", 0.9, "lower"),
            ("a", 0.9, "upper"),
            ("a", 0.5, "lower"),
            ("a", 0.5, "upper"),
        ]
    )
    y_pred = pd.DataFrame(
        [
            [0.5, 1.5, 0.75, 1.25],
            [1.0, 3.0, 1.5, 2.5],
            [2.0, 5.0, 2.5, 3.5],
            [3.0, 7.0, 3.5, 4.5],
        ],
        columns=cols,
        index=y_true.index,
    )
    return y_true, y_pred


def test_evaluate_by_index_fallback_interval_metric():
    """Jackknife fallback of evaluate_by_index works for metrics with only _evaluate.

    For a metric that is a mean over rows, the jackknife pseudo-values are
    exactly the per-row values, here the per-row interval widths.
    """
    y_true, y_pred = _make_interval_data()

    metric = _MeanIntervalWidth(multioutput="raw_values", score_average=False)
    assert metric.evaluate(y_true, y_pred).tolist() == [2.5, 0.875]

    res = metric.evaluate_by_index(y_true, y_pred)
    expected = IntervalWidth(multioutput="raw_values", score_average=False)
    expected = expected.evaluate_by_index(y_true, y_pred)

    assert isinstance(res, pd.DataFrame)
    pd.testing.assert_index_equal(res.index, y_true.index)
    res = res.sort_index(axis=1)
    expected = expected.sort_index(axis=1)
    assert res.columns.tolist() == expected.columns.tolist()
    np.testing.assert_allclose(res.to_numpy(), expected.to_numpy())

    # averaging over scores and variables works as for other metrics
    res_avg = _MeanIntervalWidth().evaluate_by_index(y_true, y_pred)
    np.testing.assert_allclose(res_avg.to_numpy(), [1.0, 2.0, 3.0, 4.0])


def test_evaluate_by_index_fallback_distr_metric():
    """Jackknife fallback of evaluate_by_index works for distribution metrics."""
    y_true = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}, index=[10, 11, 12, 13])
    y_pred = Normal(
        mu=[[1.5], [1.0], [3.5], [3.0]], sigma=1.0, index=y_true.index, columns=["a"]
    )

    res = _MeanCRPS(multioutput="raw_values").evaluate_by_index(y_true, y_pred)
    expected = CRPS(multioutput="raw_values").evaluate_by_index(y_true, y_pred)

    pd.testing.assert_index_equal(res.index, y_true.index)
    np.testing.assert_allclose(res.to_numpy(), expected.to_numpy())

    res_avg = _MeanCRPS().evaluate(y_true, y_pred)
    assert res_avg == pytest.approx(CRPS().evaluate(y_true, y_pred))


def test_metric_without_evaluate_methods_raises():
    """A metric implementing neither _evaluate nor _evaluate_by_index raises."""

    class _Empty(BaseProbaMetric):
        _tags = {"scitype:y_pred": "pred_interval"}

    y_true, y_pred = _make_interval_data()
    with pytest.raises(RecursionError, match="must implement one of"):
        _Empty().evaluate(y_true, y_pred)
