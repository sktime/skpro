"""Tests for probabilistic metrics for distribution predictions."""

import numpy as np
import pandas as pd
import pytest
from skbase.testing import QuickTester
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from skpro.distributions import Normal
from skpro.metrics._classes import SquaredDistrLoss
from skpro.regression.residual import ResidualDouble
from skpro.tests.test_all_estimators import BaseFixtureGenerator, PackageConfig

TEST_DISTS = [Normal]

# Module-level fixtures - built once, reused across all parametrized cases.
# Mirrors the setup in the deleted test_probabilistic_metrics.py.

X, _y = load_diabetes(return_X_y=True, as_frame=True)
_y = pd.DataFrame(_y)
X_train, X_test, y_train, y_test = train_test_split(X, _y, random_state=42)

# univariate regressor
_reg = ResidualDouble.create_test_instance()
_reg.fit(X_train, y_train)

_quant_uni_s = _reg.predict_quantiles(X_test, alpha=[0.5])
_quant_uni_m = _reg.predict_quantiles(X_test, alpha=[0.05, 0.5, 0.95])
_intv_uni_s = _reg.predict_interval(X_test, coverage=0.9)
_intv_uni_m = _reg.predict_interval(X_test, coverage=[0.7, 0.8, 0.9, 0.99])

y_test_uni = y_test

# multivariate predictions — simulated via column concat (same approach as original)
_y_train2 = y_train.copy()
_y_train2.columns = ["foo"]
_reg2 = ResidualDouble.create_test_instance()
_reg2.fit(X_train, _y_train2)

_quant_multi_s = pd.concat(
    [_reg2.predict_quantiles(X_test, alpha=[0.5]), _quant_uni_s], axis=1
)
_quant_multi_m = pd.concat(
    [_reg2.predict_quantiles(X_test, alpha=[0.05, 0.5, 0.95]), _quant_uni_m], axis=1
)
_intv_multi_s = pd.concat(
    [_reg2.predict_interval(X_test, coverage=0.9), _intv_uni_s], axis=1
)
_intv_multi_m = pd.concat(
    [_reg2.predict_interval(X_test, coverage=[0.7, 0.8, 0.9, 0.99]), _intv_uni_m],
    axis=1,
)

_y_test2 = y_test.copy()
_y_test2.columns = ["foo"]
y_test_multi = pd.concat([_y_test2, y_test], axis=1)

# parametrize lists — (y_true, y_pred) pairs
_ALL_CASES = list(
    zip(
        [y_test_uni] * 4 + [y_test_multi] * 4,
        [
            _quant_uni_s,
            _intv_uni_s,
            _quant_uni_m,
            _intv_uni_m,
            _quant_multi_s,
            _intv_multi_s,
            _quant_multi_m,
            _intv_multi_m,
        ],
    )
)

_QUANT_CASES = list(
    zip(
        [y_test_uni] * 2 + [y_test_multi] * 2,
        [_quant_uni_s, _quant_uni_m, _quant_multi_s, _quant_multi_m],
    )
)


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

    @pytest.mark.parametrize("y_true,y_pred", _ALL_CASES)
    @pytest.mark.parametrize("multioutput", ["uniform_average", "raw_values"])
    @pytest.mark.parametrize("score_average", [True, False])
    def test_quantile_interval_output_shape_and_type(
        self, object_instance, score_average, multioutput, y_true, y_pred
    ):
        """Test output type and shape for all uni/multi and single/multi-score cases.

        Migrated from ``test_output`` in the deleted test_probabilistic_metrics.py.
        Covers all combinations of:
        - univariate/multivariate predictions
        - single score/multi score
        - score_average True/False
        - multioutput uniform_average/raw_values
        """

        metric = object_instance
        tag = metric.get_tag("scitype:y_pred", None, raise_error=False)

        # only run for interval/quantile metrics
        if tag not in ["pred_quantiles", "pred_interval"]:
            pytest.skip("metric is not quantile or interval type")

        loss = metric.set_params(score_average=score_average, multioutput=multioutput)

        eval_loss = loss(y_true, y_pred)
        index_loss = loss.evaluate_by_index(y_true, y_pred)

        no_vars = len(y_pred.columns.get_level_values(0).unique())
        no_scores = len(y_pred.columns.get_level_values(1).unique())

        # adjust score count when interval metric receives quantile-format input
        if (
            0.5 in y_pred.columns.get_level_values(1)
            and tag == "pred_interval"
            and y_pred.columns.nlevels == 2
        ):
            no_scores -= 1
            no_scores //= 2
            if no_scores == 0:
                no_vars = 0

        if score_average and multioutput == "uniform_average":
            assert isinstance(eval_loss, float)
            assert isinstance(index_loss, pd.Series)
            assert len(index_loss) == y_pred.shape[0]

        if not score_average and multioutput == "uniform_average":
            assert isinstance(eval_loss, pd.Series)
            assert isinstance(index_loss, pd.DataFrame)

            if tag == "pred_quantiles" and y_pred.columns.nlevels == 3:
                assert len(eval_loss) == 2 * no_scores
            else:
                assert len(eval_loss) == no_scores

        if not score_average and multioutput == "raw_values":
            assert isinstance(eval_loss, pd.Series)
            assert isinstance(index_loss, pd.DataFrame)

            true_len = no_vars * no_scores
            if tag == "pred_quantiles" and y_pred.columns.nlevels == 3:
                assert len(eval_loss) == 2 * true_len
            else:
                assert len(eval_loss) == true_len

        if score_average and multioutput == "raw_values":
            assert isinstance(eval_loss, pd.Series)
            assert isinstance(index_loss, pd.DataFrame)
            assert len(eval_loss) == no_vars

        # row count always equals number of test samples
        assert index_loss.shape[0] == y_pred.shape[0]

    # Migrated from test_probabilistic_metrics.py — alpha validation
    @pytest.mark.parametrize("y_true,y_pred", _QUANT_CASES)
    def test_quantile_alpha_positive(self, object_instance, y_true, y_pred):
        """Test output when the required quantile is present in y_pred

        Migrated from ``test_evaluate_alpha_positive`` in the deleted file
        """
        metric = object_instance
        tag = metric.get_tag("scitype:y_pred", None, raise_error=False)

        if tag != "pred_quantiles":
            pytest.skip("not a quantile metric")

        # alpha=0.5 is present in all quantile test cases — must not raise
        loss = metric.set_params(alpha=0.5, score_average=False)
        res = loss(y_true=y_true, y_pred=y_pred)
        assert len(res) == 1

        # if 0.95 is also present, test multi-alpha selection
        if all(q in y_pred.columns.get_level_values(1) for q in [0.5, 0.95]):
            loss2 = metric.set_params(alpha=[0.5, 0.95], score_average=False)
            res2 = loss2(y_true=y_true, y_pred=y_pred)
            assert len(res2) == 2

    @pytest.mark.parametrize("y_true,y_pred", _QUANT_CASES)
    def test_quantile_alpha_negative(self, object_instance, y_true, y_pred):
        """Test that a missing alpha raises ValueError

        Migrated from ``test_evaluate_alpha_negative`` in the deleted file
        """
        metric = object_instance
        tag = metric.get_tag("scitype:y_pred", None, raise_error=False)

        if tag != "pred_quantiles":
            pytest.skip("not a quantile metric")

        with pytest.raises(ValueError):
            # 0.3 not present in any test quantile data - must raise
            loss = metric.set_params(alpha=0.3)
            loss(y_true=y_true, y_pred=y_pred)


def test_squared_distr_loss_uses_pdf():
    """SquaredDistrLoss must compute -2*pdf(y) + pdfnorm(a=2), not -2*log_pdf(y).

    The squared distribution loss (Gneiting/Brier score) is:

        L(y,d)= -2*p_d(y) + ||p_d||^2

    where p_d(y) is the probability density, NOT the log-density.

    pdfnorm uses a Monte Carlo approximation, so we cannot compare two
    independent calls numerically. Instead, two behavioural properties that
    definitively distinguish the correct formula from the log_pdf bug are tested:

    1. Sign: at the mode of Normal(0, 1), the correct loss is negative
       (-2 * pdf(0) ≈ -0.798 dominates pdfnorm ≈ 0.282).
       The log_pdf formula gives +2.12 (positive) - a clear sign flip.

    2. Magnitude gap: the correct result (~-0.516) and the buggy result (~+2.12)
       differ by more than 2.5 units; any mix-up is caught with a 1.0-unit gap.
    """
    mu, sigma = 0.0, 1.0
    y_true = pd.DataFrame([[mu]])
    y_pred = Normal(mu=[[mu]], sigma=[[sigma]])

    loss = SquaredDistrLoss()
    result = loss._evaluate_by_index(y_true, y_pred)

    assert isinstance(result, pd.DataFrame), "Result must be a DataFrame"
    obtained = result.values[0, 0]

    # Check 1 - sign: correct formula at mode is negative; log_pdf formula is > 0.
    assert obtained < 0, (
        f"SquaredDistrLoss at mode of Normal(0,1) must be negative, "
        f"got {obtained:.6f}. "
        "A positive value indicates log_pdf is incorrectly used instead of pdf."
    )

    # Check 2 - gap from log_pdf variant: correct≈-0.516, buggy≈+2.12.
    log_pdf_val = y_pred.log_pdf(y_true).values[0, 0]
    pdfnorm_val = y_pred.pdfnorm(a=2).values[0, 0]
    buggy = -2 * log_pdf_val + pdfnorm_val
    assert abs(obtained - buggy) > 1.0, (
        f"SquaredDistrLoss ({obtained:.4f}) is suspiciously close to the "
        f"log_pdf formula result ({buggy:.4f}). Check the loss formula."
    )


def test_squared_distr_loss_multivariate():
    # SquaredDistrLoss in multivariate mode must return a single-column DataFrame
    y_pred = Normal(mu=[[0.0, 1.0], [2.0, 3.0]], sigma=1.0)
    y_true = y_pred.sample()

    loss = SquaredDistrLoss(multivariate=True)
    result = loss._evaluate_by_index(y_true, y_pred)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["density"]
    assert result.shape == (2, 1)
