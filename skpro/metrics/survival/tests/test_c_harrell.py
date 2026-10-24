"""Tests for Harell's C-index."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import pandas as pd
import pytest


@pytest.mark.parametrize("concordant", [True, False])
@pytest.mark.parametrize("pass_c", ["True", "False", "None"])
@pytest.mark.parametrize("normalization", ["overall", "index"])
def test_charrell_logic(concordant, pass_c, normalization):
    """Test the logic of the Harrell's C-index metric.

    Parameters
    ----------
    concordant : bool, optional, default=True
        If True, the test examples are fully concordant.
        If False, the test examples are fully discordant.
    pass_c : bool, optional, default=True
        If True, the ``C_true`` argument is passed to the metric, with censoring data.
        If None, the ``C_true`` argument is passed to the metric, with value None.
        If False, the ``C_true`` argument is not passed to the metric.
    normalization : str, optional, default="overall"
        The normalization method for the metric.
    """
    from skpro.distributions import Normal
    from skpro.metrics.survival._c_harrell import ConcordanceHarrell

    # examples are constructed to be fully concordant or discordant,
    # depending on the value of `concordant`
    y_true = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 4, 3, 2]})
    c_true = pd.DataFrame({"a": [1, 0, 1, 0], "b": [0, 1, 0, 1]})
    y_pred_mean = pd.DataFrame({"a": [2, 3, 4, 5], "b": [6, 5, 4, 3]})

    if not concordant:
        y_pred_mean = -y_pred_mean
    y_pred = Normal(y_pred_mean, sigma=1, columns=pd.Index(["a", "b"]))

    # evaluate the metric
    metric = ConcordanceHarrell(normalization=normalization, tie_score=int(concordant))
    metric_args = {"y_true": y_true, "y_pred": y_pred}
    if pass_c == "True":
        metric_args["C_true"] = c_true
    elif pass_c == "None":
        metric_args["C_true"] = c_true

    res = metric(**metric_args)
    res_by_index = metric.evaluate_by_index(**metric_args)

    assert res_by_index.shape == y_true.shape

    # test assumptions
    # if concordant, the result should be 1
    # if discordant, the result should be 0
    assert res == concordant

    if normalization == "index":
        assert (res_by_index == concordant).all().all()


def test_charrell_score_options():
    """Test score parameter options for ConcordanceHarrell."""
    from skpro.distributions import Normal
    from skpro.metrics.survival import ConcordanceHarrell

    y_true = pd.DataFrame({"time": [2.0, 4.0, 6.0, 8.0]})
    y_pred = Normal(mu=[[3.0], [4.5], [7.0], [5.0]], sigma=1.0, columns=["time"])

    # score='median' should work and produce the same result as ppf at 0.5
    res_median = ConcordanceHarrell(score="median")(y_true, y_pred)
    res_ppf = ConcordanceHarrell(score="ppf", score_args={"p": 0.5})(y_true, y_pred)
    res_quantile = ConcordanceHarrell(
        score="quantile", score_args={"alpha": 0.5}
    )(y_true, y_pred)
    res_mean = ConcordanceHarrell(score="mean")(y_true, y_pred)

    assert res_median == res_ppf
    assert res_median == res_quantile
    assert res_median == res_mean


def test_charrell_invalid_score():
    """Test that invalid score method raises an informative AttributeError."""
    from skpro.distributions import Normal
    from skpro.metrics.survival import ConcordanceHarrell

    y_true = pd.DataFrame({"time": [2.0, 4.0, 6.0, 8.0]})
    y_pred = Normal(mu=[[3.0], [4.5], [7.0], [5.0]], sigma=1.0, columns=["time"])

    with pytest.raises(AttributeError, match="Valid score options include"):
        ConcordanceHarrell(score="nonexistent_score")(y_true, y_pred)


def test_base_distribution_median():
    """Test median method on BaseDistribution."""
    from skpro.distributions import Normal

    dist = Normal(mu=[[1.0], [2.0]], sigma=1.0, columns=["val"])
    assert hasattr(dist, "median")
    pd.testing.assert_frame_equal(dist.median(), dist.ppf(0.5))
