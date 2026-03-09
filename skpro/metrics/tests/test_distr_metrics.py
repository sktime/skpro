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

TEST_DISTS = [Normal]


class TestAllDistrMetrics(PackageConfig, BaseFixtureGenerator, QuickTester):
    """Generic tests for all probabilistic regression metrics in the package."""

    # class variables which can be overridden by descendants
    # ------------------------------------------------------

    # which object types are generated; None=all, or scitype string
    # passed to skpro.registry.all_objects as object_type
    object_type_filter = "metric"

    def _all_objects(self):
        from skpro.registry import all_objects

        return all_objects(
            "metric", return_names=False, exclude_objects=self.exclude_objects
        )

    @pytest.mark.parametrize("dist", TEST_DISTS)
    @pytest.mark.parametrize("pass_c", [True, False])
    @pytest.mark.parametrize("multivariate", [True, False])
    @pytest.mark.parametrize("multioutput", ["raw_values", "uniform_average"])
    @pytest.mark.parametrize("score_average", [True, False])
    def test_distr_evaluate(
        self, object_instance, dist, pass_c, multivariate, multioutput, score_average
    ):
        """Test expected output of evaluate functions."""
        metric = object_instance
        scitype = metric.get_tag("scitype:y_pred")

        params = {"multioutput": multioutput}
        if "score_average" in metric.get_params():
            params["score_average"] = score_average

        m = metric.set_params(**params)

        if "multivariate" in metric.get_params():
            m = m.set_params(multivariate=multivariate)

        if scitype == "pred_proba":
            y_pred = dist.create_test_instance()
            y_true = y_pred.sample()

            metric_args = {"y_true": y_true, "y_pred": y_pred}
            if pass_c:
                c_true = np.random.randint(0, 2, size=y_true.shape)
                c_true = pd.DataFrame(
                    c_true, columns=y_true.columns, index=y_true.index
                )
                metric_args["c_true"] = c_true

            res = m.evaluate_by_index(**metric_args)

            # For pred_proba, we expect DataFrame usually
            if isinstance(res, pd.Series):
                res = res.to_frame()

            assert isinstance(res, pd.DataFrame)
            # assert (res.columns == expected_cols).all()
            # assert res.shape == (y_true.shape[0], len(expected_cols))

            res = m.evaluate(**metric_args)

            expect_df = not multivariate and multioutput == "raw_values"
            if expect_df:
                assert isinstance(res, pd.DataFrame)
                # assert (res.columns == expected_cols).all()
                # assert res.shape == (1, len(expected_cols))
            else:
                assert isinstance(res, float)

        elif scitype in ["pred_quantiles", "pred_interval"]:
            # Generate data
            X, y = load_diabetes(return_X_y=True, as_frame=True)
            y = pd.DataFrame(y)
            X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

            if multivariate:
                y_train.columns = ["foo"]
                y_test.columns = ["foo"]
                y_train = pd.concat(
                    [y_train, y_train.copy().rename(columns={"foo": "bar"})], axis=1
                )
                y_test = pd.concat(
                    [y_test, y_test.copy().rename(columns={"foo": "bar"})], axis=1
                )

            reg = ResidualDouble.create_test_instance()
            reg.fit(X_train, y_train)

            if scitype == "pred_quantiles":
                y_pred = reg.predict_quantiles(X_test, alpha=[0.05, 0.5, 0.95])
                if "alpha" in m.get_params():
                    m = m.set_params(alpha=None)
            else:
                y_pred = reg.predict_interval(X_test, coverage=[0.9])

            y_true = y_test
            metric_args = {"y_true": y_true, "y_pred": y_pred}

            eval_loss = m.evaluate(**metric_args)
            index_loss = m.evaluate_by_index(**metric_args)

            no_vars = len(y_pred.columns.get_level_values(0).unique())
            no_scores = len(y_pred.columns.get_level_values(1).unique())

            if (
                0.5 in y_pred.columns.get_level_values(1)
                and scitype == "pred_interval"
                and y_pred.columns.nlevels == 2
            ):
                no_scores = no_scores - 1
                no_scores = no_scores / 2
                if no_scores == 0:
                    no_vars = 0

            if score_average and multioutput == "uniform_average":
                assert isinstance(eval_loss, float)
                assert isinstance(index_loss, pd.Series)
                assert len(index_loss) == y_pred.shape[0]

            if not score_average and multioutput == "uniform_average":
                assert isinstance(eval_loss, pd.Series)
                assert isinstance(index_loss, pd.DataFrame)

                if scitype == "pred_quantiles" and y_pred.columns.nlevels == 3:
                    assert len(eval_loss) == 2 * no_scores
                else:
                    assert len(eval_loss) == no_scores

            if not score_average and multioutput == "raw_values":
                assert isinstance(eval_loss, pd.Series)
                assert isinstance(index_loss, pd.DataFrame)

                true_len = no_vars * no_scores

                if scitype == "pred_quantiles" and y_pred.columns.nlevels == 3:
                    assert len(eval_loss) == 2 * true_len
                else:
                    assert len(eval_loss) == true_len

            if score_average and multioutput == "raw_values":
                assert isinstance(eval_loss, pd.Series)
                assert isinstance(index_loss, pd.DataFrame)

                assert len(eval_loss) == no_vars

    def test_evaluate_alpha_positive(self, object_instance):
        """Tests output when required quantile is present."""
        metric = object_instance
        if metric.get_tag("scitype:y_pred") != "pred_quantiles":
            return

        X, y = load_diabetes(return_X_y=True, as_frame=True)
        y = pd.DataFrame(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        reg = ResidualDouble.create_test_instance()
        reg.fit(X_train, y_train)

        # Case 1: alpha=0.5
        y_pred = reg.predict_quantiles(X_test, alpha=[0.5])
        Loss = metric.set_params(alpha=0.5, score_average=False)
        res = Loss(y_true=y_test, y_pred=y_pred)
        assert len(res) == 1

        # Case 2: alpha=[0.5, 0.95]
        y_pred = reg.predict_quantiles(X_test, alpha=[0.5, 0.95])
        Loss = metric.set_params(alpha=[0.5, 0.95], score_average=False)
        res = Loss(y_true=y_test, y_pred=y_pred)
        assert len(res) == 2

    def test_evaluate_alpha_negative(self, object_instance):
        """Tests whether correct error raised when required quantile not present."""
        metric = object_instance
        if metric.get_tag("scitype:y_pred") != "pred_quantiles":
            return

        X, y = load_diabetes(return_X_y=True, as_frame=True)
        y = pd.DataFrame(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

        reg = ResidualDouble.create_test_instance()
        reg.fit(X_train, y_train)

        y_pred = reg.predict_quantiles(X_test, alpha=[0.5])

        with pytest.raises(ValueError):
            # 0.3 not in test quantile data so raise error.
            Loss = metric.set_params(alpha=0.3)
            Loss(y_true=y_test, y_pred=y_pred)
