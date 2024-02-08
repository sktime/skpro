"""Tests for probabilistic metrics for distribution predictions."""
import warnings

import pandas as pd
import pytest

from skpro.distributions import Normal
from skpro.metrics import (
    CRPS,
    SPLL,
    ConcordanceHarrell,
    LinearizedLogLoss,
    LogLoss,
    SquaredDistrLoss,
)

warnings.filterwarnings("ignore", category=FutureWarning)

DISTR_METRICS = [
    CRPS,
    SPLL,
    ConcordanceHarrell,
    LinearizedLogLoss,
    LogLoss,
    SquaredDistrLoss,
]

normal_dists = [Normal]


@pytest.mark.parametrize("normal", normal_dists)
@pytest.mark.parametrize("metric", DISTR_METRICS)
@pytest.mark.parametrize("multivariate", [True, False])
@pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average"])
def test_distr_evaluate(normal, metric, multivariate, multioutput):
    """Test expected output of evaluate functions."""
    y_pred = normal.create_test_instance()
    y_true = y_pred.sample()

    m = metric(multivariate=multivariate, multioutput=multioutput)

    if not multivariate:
        expected_cols = y_true.columns
    else:
        expected_cols = ["score"]

    res = m.evaluate_by_index(y_true, y_pred)
    assert isinstance(res, pd.DataFrame)
    assert (res.columns == expected_cols).all()
    assert res.shape == (y_true.shape[0], len(expected_cols))

    res = m.evaluate(y_true, y_pred)

    expect_df = not multivariate and multioutput == "raw_values"
    if expect_df:
        assert isinstance(res, pd.DataFrame)
        assert (res.columns == expected_cols).all()
        assert res.shape == (1, len(expected_cols))
    else:
        assert isinstance(res, float)
