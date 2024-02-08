"""Tests for probabilistic metrics for distribution predictions."""
import warnings

import pandas as pd
import pytest
from skbase.testing import QuickTester

from skpro.distributions import Normal
from skpro.tests.test_all_estimators import BaseFixtureGenerator, PackageConfig


TEST_DISTS = [Normal]


class TestAllDistrMetrics(PackageConfig, BaseFixtureGenerator, QuickTester):
    """Generic tests for all regressors in the mini package."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # which object types are generated; None=all, or scitype string
    # passed to skpro.registry.all_objects as object_type
    object_type_filter = "metric_distr"

    @pytest.mark.parametrize("normal", TEST_DISTS)
    @pytest.mark.parametrize("multivariate", [True, False])
    @pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average"])
    def test_distr_evaluate(object_instance, dist, multivariate, multioutput):
        """Test expected output of evaluate functions."""
        metric = object_instance

        y_pred = dist.create_test_instance()
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
