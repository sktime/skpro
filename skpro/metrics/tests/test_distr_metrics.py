"""Tests for probabilistic metrics for distribution predictions."""
import numpy as np
import pandas as pd
import pytest
from skbase.testing import QuickTester

from skpro.distributions import Normal
from skpro.metrics._classes import SquaredDistrLoss
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


def test_squared_distr_loss_uses_pdf():
    """SquaredDistrLoss must compute -2*pdf(y) + pdfnorm(a=2), not -2*log_pdf(y).

    The squared distribution loss (Gneiting/Brier score) is:

        L(y, d) = -2 * p_d(y) + ||p_d||^2

    where p_d(y) is the probability density, NOT the log-density.

    pdfnorm uses a Monte Carlo approximation, so we cannot compare two independent
    calls numerically. Instead, two behavioural properties that definitively
    distinguish the correct formula from the log_pdf bug are tested:

    1. Sign: at the mode of Normal(0, 1), the correct loss is negative
       (-2 * pdf(0) ≈ -0.798 dominates pdfnorm ≈ 0.282).
       The log_pdf formula gives +2.12 (positive) — a clear sign flip.

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

    # Check 1 — sign: correct formula at mode is negative; log_pdf formula is > 0.
    assert obtained < 0, (
        f"SquaredDistrLoss at mode of Normal(0,1) must be negative, "
        f"got {obtained:.6f}. "
        "A positive value indicates log_pdf is incorrectly used instead of pdf."
    )

    # Check 2 — gap from log_pdf variant: correct ≈ -0.516, buggy ≈ +2.12.
    log_pdf_val = y_pred.log_pdf(y_true).values[0, 0]
    pdfnorm_val = y_pred.pdfnorm(a=2).values[0, 0]
    buggy = -2 * log_pdf_val + pdfnorm_val
    assert abs(obtained - buggy) > 1.0, (
        f"SquaredDistrLoss ({obtained:.4f}) is suspiciously close to the "
        f"log_pdf formula result ({buggy:.4f}). Check the loss formula."
    )


def test_squared_distr_loss_multivariate():
    """SquaredDistrLoss in multivariate mode must return a single-column DataFrame."""
    y_pred = Normal(mu=[[0.0, 1.0], [2.0, 3.0]], sigma=1.0)
    y_true = y_pred.sample()

    loss = SquaredDistrLoss(multivariate=True)
    result = loss._evaluate_by_index(y_true, y_pred)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["density"]
    assert result.shape == (2, 1)
