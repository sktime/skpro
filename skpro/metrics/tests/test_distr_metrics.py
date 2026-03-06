"""Tests for probabilistic metrics for distribution predictions."""
import numpy as np
import pandas as pd
import pytest
from skbase.testing import QuickTester
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from skpro.distributions import Normal
from skpro.regression.residual import ResidualDouble
from skpro.tests.test_all_estimators import BaseFixtureGenerator, PackageConfig

X_DIAB, y_DIAB = load_diabetes(return_X_y=True, as_frame=True)
y_DIAB = pd.DataFrame(y_DIAB)
X_TRAIN, X_TEST, y_TRAIN, y_TEST = train_test_split(X_DIAB, y_DIAB, random_state=42)

REG_QUANT_INT = ResidualDouble.create_test_instance()
REG_QUANT_INT.fit(X_TRAIN, y_TRAIN)


def _make_metric_test_data(scitype_y_pred):
    """Create metric test data matching the metric's prediction scitype."""
    if scitype_y_pred == "pred_proba":
        y_pred = Normal.create_test_instance()
        y_true = y_pred.sample()
        return y_true, y_pred

    if scitype_y_pred == "pred_quantiles":
        y_true = y_TEST.copy()
        y_pred = REG_QUANT_INT.predict_quantiles(X_TEST, alpha=[0.1, 0.5, 0.9])
        return y_true, y_pred

    if scitype_y_pred == "pred_interval":
        y_true = y_TEST.copy()
        y_pred = REG_QUANT_INT.predict_interval(X_TEST, coverage=[0.8, 0.9])
        return y_true, y_pred

    raise ValueError(f"Unsupported scitype:y_pred for test data: {scitype_y_pred}")


class TestAllDistrMetrics(PackageConfig, BaseFixtureGenerator, QuickTester):
    """Generic tests for all probabilistic regression metrics in the package."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # which object types are generated; None=all, or scitype string
    # passed to skpro.registry.all_objects as object_type
    object_type_filter = "metric"

    @pytest.mark.parametrize("pass_sample_weight", [True, False])
    @pytest.mark.parametrize("score_average", [True, False])
    @pytest.mark.parametrize("multivariate", [True, False])
    @pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average"])
    def test_distr_evaluate(
        self,
        object_instance,
        pass_sample_weight,
        score_average,
        multivariate,
        multioutput,
    ):
        """Test evaluate/evaluate_by_index contract including weighted evaluate."""
        metric = object_instance

        if metric.get_tag("capability:survival", False, raise_error=False):
            pytest.skip("Survival metrics require dedicated fixtures and C_true setup.")

        scitype_y_pred = metric.get_tag("scitype:y_pred")
        y_true, y_pred = _make_metric_test_data(scitype_y_pred)

        m = metric.set_params(multioutput=multioutput)

        if "score_average" in metric.get_params():
            m = m.set_params(score_average=score_average)

        if "multivariate" in metric.get_params():
            m = m.set_params(multivariate=multivariate)
        elif multivariate:
            pytest.skip("Metric does not expose multivariate parameter.")

        metric_args = {"y_true": y_true, "y_pred": y_pred}
        if pass_sample_weight:
            n_rows = len(y_true)
            metric_args["sample_weight"] = np.arange(1, n_rows + 1)

        # evaluate_by_index must preserve sample axis and index alignment
        res_ix = m.evaluate_by_index(y_true=y_true, y_pred=y_pred)
        assert isinstance(res_ix, (pd.Series, pd.DataFrame))
        assert len(res_ix) == len(y_true)

        # evaluate must support both weighted and unweighted contract paths
        res = m.evaluate(y_true=y_true, y_pred=y_pred)
        res_w = m.evaluate(**metric_args)

        assert isinstance(res, (float, np.floating, pd.Series))
        assert isinstance(res_w, (float, np.floating, pd.Series))

        if isinstance(res, pd.Series):
            assert isinstance(res_w, pd.Series)
            assert (res.index == res_w.index).all()
        else:
            assert np.isscalar(res_w)


def test_sample_weight_logloss():
    from scipy.stats import norm

    from skpro.metrics import LogLoss

    y_true = pd.DataFrame({"y": [0, 0, 0]})

    mu = np.array([[0], [10], [0]])
    sigma = np.array([[1], [1], [1]])
    y_pred = Normal(mu, sigma, index=y_true.index, columns=y_true.columns)

    metric = LogLoss()
    loss = metric(y_true, y_pred)

    L1 = -norm.logpdf(0, 0, 1)
    L2 = -norm.logpdf(0, 10, 1)
    expected_mean = (2 * L1 + L2) / 3
    assert np.isclose(loss, expected_mean)

    weights = [1, 0, 1]
    loss_w = metric(y_true, y_pred, sample_weight=weights)
    assert np.isclose(loss_w, L1)


def test_multioutput_weights_logloss():
    from scipy.stats import norm

    from skpro.metrics import LogLoss

    y_true = pd.DataFrame({"y1": [0], "y2": [0]})

    mu = np.array([[0, 10]])
    sigma = np.array([[1, 1]])
    y_pred = Normal(mu, sigma, index=y_true.index, columns=y_true.columns)

    metric = LogLoss(multioutput=[0.1, 0.9])
    loss = metric(y_true, y_pred)

    L1 = -norm.logpdf(0, 0, 1)
    L2 = -norm.logpdf(0, 10, 1)

    expected_loss = 0.1 * L1 + 0.9 * L2
    assert np.isclose(loss, expected_loss)
