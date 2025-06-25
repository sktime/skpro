"""Merged tests for probabilistic quantile, interval, and distribution metrics."""
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from skbase.testing import QuickTester
from skpro.distributions import Normal
from skpro.metrics._classes import (
    ConstraintViolation,
    EmpiricalCoverage,
    IntervalWidth,
    PinballLoss,
)
from skpro.regression.residual import ResidualDouble
from skpro.tests.test_all_estimators import BaseFixtureGenerator, PackageConfig


class TestAllDistrMetrics(PackageConfig, BaseFixtureGenerator, QuickTester):
    """Generic tests for all probabilistic regression metrics in the package."""

    object_type_filter = "metric_distr"

    TEST_DISTS = [Normal]

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

    @pytest.mark.parametrize("metric", [PinballLoss, EmpiricalCoverage, IntervalWidth, ConstraintViolation])
    @pytest.mark.parametrize("score_average", [True, False])
    @pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
    def test_prob_metrics_output(self, metric, score_average, multioutput):
        """Test output correctness for quantile and interval metrics."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        y = pd.DataFrame(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        reg = ResidualDouble.create_test_instance()
        reg.fit(X_train, y_train)

        quantile_pred = reg.predict_quantiles(X_test, alpha=[0.05, 0.5, 0.95])
        interval_pred = reg.predict_interval(X_test, coverage=[0.7, 0.8, 0.9, 0.99])

        for y_pred in [quantile_pred, interval_pred]:
            y_true = y_test
            loss = metric.create_test_instance()
            loss.set_params(score_average=score_average, multioutput=multioutput)

            eval_loss = loss(y_true, y_pred)
            index_loss = loss.evaluate_by_index(y_true, y_pred)

            no_vars = len(y_pred.columns.get_level_values(0).unique())
            no_scores = len(y_pred.columns.get_level_values(1).unique())

            if (
                0.5 in y_pred.columns.get_level_values(1)
                and loss.get_tag("scitype:y_pred") == "pred_interval"
                and y_pred.columns.nlevels == 2
            ):
                no_scores -= 1
                no_scores /= 2  # one interval loss per two quantiles given
                if no_scores == 0:  # if only 0.5 quant, no output to interval loss
                    no_vars = 0

            if score_average and multioutput == "uniform_average":
                assert isinstance(eval_loss, float)
                assert isinstance(index_loss, pd.Series)
                assert len(index_loss) == y_pred.shape[0]

            if not score_average and multioutput == "uniform_average":
                assert isinstance(eval_loss, pd.Series)
                assert isinstance(index_loss, pd.DataFrame)
                if (
                    loss.get_tag("scitype:y_pred") == "pred_quantiles"
                    and y_pred.columns.nlevels == 3
                ):
                    assert len(eval_loss) == 2 * no_scores
                else:
                    assert len(eval_loss) == no_scores

            if not score_average and multioutput == "raw_values":
                assert isinstance(eval_loss, pd.Series)
                assert isinstance(index_loss, pd.DataFrame)
                true_len = no_vars * no_scores
                if (
                    loss.get_tag("scitype:y_pred") == "pred_quantiles"
                    and y_pred.columns.nlevels == 3
                ):
                    assert len(eval_loss) == 2 * true_len
                else:
                    assert len(eval_loss) == true_len

            if score_average and multioutput == "raw_values":
                assert isinstance(eval_loss, pd.Series)
                assert isinstance(index_loss, pd.DataFrame)
                assert len(eval_loss) == no_vars

    @pytest.mark.parametrize("Metric", [PinballLoss])
    @pytest.mark.parametrize(
        "y_pred, y_true",
        [("quantile", "test_uni"), ("quantile", "test_multi")],
    )
    def test_evaluate_alpha(self, Metric, y_pred, y_true):
        """Tests behavior when required quantiles are present or missing."""
        X, y = load_diabetes(return_X_y=True, as_frame=True)
        y = pd.DataFrame(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        reg = ResidualDouble.create_test_instance()
        reg.fit(X_train, y_train)

        y_pred_quantiles = reg.predict_quantiles(X_test, alpha=[0.5, 0.95])

        Loss = Metric.create_test_instance().set_params(alpha=0.5, score_average=False)
        res = Loss(y_true=y_test, y_pred=y_pred_quantiles)
        assert len(res) == 1

        with pytest.raises(ValueError):
            Loss = Metric.create_test_instance().set_params(alpha=0.3)
            Loss(y_true=y_test, y_pred=y_pred_quantiles)

        if all(x in y_pred_quantiles.columns.get_level_values(1) for x in [0.5, 0.95]):
            Loss = Metric.create_test_instance().set_params(
                alpha=[0.5, 0.95], score_average=False
            )
            res = Loss(y_true=y_test, y_pred=y_pred_quantiles)
            assert len(res) == 2
