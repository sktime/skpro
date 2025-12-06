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


def test_sample_weight_logloss():
    from scipy.stats import norm

    from skpro.metrics import LogLoss

    y_true = pd.DataFrame({"y": [0, 0, 0]})
    # Normal(0, 1). log_pdf(0) = -0.5*log(2pi) - 0.5*0 = -0.9189
    # Normal(10, 1). log_pdf(0) = -0.5*log(2pi) - 0.5*100 = -50.9189

    mu = np.array([[0], [10], [0]])
    sigma = np.array([[1], [1], [1]])
    y_pred = Normal(mu, sigma, index=y_true.index, columns=y_true.columns)

    metric = LogLoss()
    loss = metric(y_true, y_pred)

    # Loss for sample 0 and 2 is L1. Loss for sample 1 is L2.
    # L1 = -(-0.9189) = 0.9189
    # L2 = -(-50.9189) = 50.9189
    # Mean = (2*L1 + L2)/3

    L1 = -norm.logpdf(0, 0, 1)
    L2 = -norm.logpdf(0, 10, 1)
    expected_mean = (2 * L1 + L2) / 3
    assert np.isclose(loss, expected_mean)

    # Weights to ignore the bad prediction (index 1)
    weights = [1, 0, 1]
    loss_w = metric(y_true, y_pred, sample_weight=weights)
    assert np.isclose(loss_w, L1)


def test_multioutput_weights_logloss():
    from scipy.stats import norm

    from skpro.metrics import LogLoss

    y_true = pd.DataFrame({"y1": [0], "y2": [0]})
    # y1: Normal(0, 1) -> Loss L1
    # y2: Normal(10, 1) -> Loss L2

    mu = np.array([[0, 10]])
    sigma = np.array([[1, 1]])
    y_pred = Normal(mu, sigma, index=y_true.index, columns=y_true.columns)

    metric = LogLoss(multioutput=[0.1, 0.9])
    loss = metric(y_true, y_pred)

    L1 = -norm.logpdf(0, 0, 1)
    L2 = -norm.logpdf(0, 10, 1)

    expected_loss = 0.1 * L1 + 0.9 * L2
    assert np.isclose(loss, expected_loss)
