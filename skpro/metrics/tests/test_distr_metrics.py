"""Tests for probabilistic metrics for distribution predictions."""
import numpy as np
import pandas as pd
import pytest
from skbase.testing import QuickTester

from skpro.distributions import Normal
from skpro.tests.test_all_estimators import BaseFixtureGenerator, PackageConfig

TEST_DISTS = [Normal]


class TestAllDistrMetrics(PackageConfig, BaseFixtureGenerator, QuickTester):
    """Generic tests for all probabilistic regression metrics in the package."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # which object types are generated; None=all, or scitype string
    # passed to skpro.registry.all_objects as object_type
    object_type_filter = "metric_distr"

    @pytest.mark.parametrize("dist", TEST_DISTS)
    @pytest.mark.parametrize("pass_c", [True, False])
    @pytest.mark.parametrize("multivariate", [True, False])
    @pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average"])
    def test_distr_evaluate(
        self, object_instance, dist, pass_c, multivariate, multioutput
    ):
        """Test expected output of evaluate functions."""
        metric = object_instance

        y_pred = dist.create_test_instance()
        y_true = y_pred.sample()

        m = metric.set_params(multioutput=multioutput)
        if "multivariate" in metric.get_params():
            m = m.set_params(multivariate=multivariate)

        if not multivariate:
            expected_cols = y_true.columns
        else:
            expected_cols = ["score"]

        metric_args = {"y_true": y_true, "y_pred": y_pred}
        if pass_c:
            c_true = np.random.randint(0, 2, size=y_true.shape)
            c_true = pd.DataFrame(c_true, columns=y_true.columns, index=y_true.index)
            metric_args["c_true"] = c_true

        res = m.evaluate_by_index(**metric_args)
        assert isinstance(res, pd.DataFrame)
        assert (res.columns == expected_cols).all()
        assert res.shape == (y_true.shape[0], len(expected_cols))

        res = m.evaluate(**metric_args)

        expect_df = not multivariate and multioutput == "raw_values"
        if expect_df:
            assert isinstance(res, pd.DataFrame)
            assert (res.columns == expected_cols).all()
            assert res.shape == (1, len(expected_cols))
        else:
            assert isinstance(res, float)
