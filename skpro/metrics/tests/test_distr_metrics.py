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

    @pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
    @pytest.mark.parametrize("score_average", [True, False])
    def test_quantile_interval_metric_output(
        self, object_instance, multioutput, score_average
    ):
        """Test output format for quantile and interval metrics."""

        metric = object_instance
        tag = metric.get_tag("scitype:y_pred", None, raise_error=False)

        # only run for interval/quantile metrics
        if tag not in ["pred_quantiles", "pred_interval"]:
            pytest.skip("metric is not quantile or interval type")

        y_true = pd.DataFrame({"y": [1, 2, 3]})

        if tag == "pred_quantiles":
            y_pred = pd.DataFrame(
                {
                    ("y", 0.1): [1, 2, 3],
                    ("y", 0.5): [1, 2, 3],
                    ("y", 0.9): [1, 2, 3],
                }
            )
        else:
            y_pred = pd.DataFrame(
                {
                    ("y", 0.9, "lower"): [0, 1, 2],
                    ("y", 0.9, "upper"): [2, 3, 4],
                }
            )

        metric = metric.set_params(
            multioutput=multioutput,
            score_average=score_average,
        )

        res = metric(y_true=y_true, y_pred=y_pred)
        res_index = metric.evaluate_by_index(y_true=y_true, y_pred=y_pred)

        assert res is not None
        assert res_index is not None

    def test_quantile_alpha_validation(self, object_instance):
        """Test that quantile metrics raise error for missing alpha."""

        metric = object_instance
        tag = metric.get_tag("scitype:y_pred", None, raise_error=False)

        if tag != "pred_quantiles":
            pytest.skip("not a quantile metric")

        y_true = pd.DataFrame({"y": [1, 2, 3]})

        y_pred = pd.DataFrame(
            {
                ("y", 0.1): [1, 2, 3],
                ("y", 0.5): [1, 2, 3],
                ("y", 0.9): [1, 2, 3],
            }
        )

        metric = metric.set_params(alpha=0.3)

        with pytest.raises(ValueError):
            metric(y_true=y_true, y_pred=y_pred)
