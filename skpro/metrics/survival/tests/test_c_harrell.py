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


def test_charrell_ties_with_censoring():
    """Tied risk scores on non-comparable censored pairs must not inflate C-index."""
    from skpro.distributions import Normal
    from skpro.metrics.survival._c_harrell import ConcordanceHarrell

    y_true = pd.DataFrame({"t": [1.0, 2.0, 3.0, 4.0]})
    c_true = pd.DataFrame({"t": [1, 1, 0, 0]})

    # Constant predictions: only (2, 3) is comparable, and it is a tie -> C = 0.5
    y_pred_const = Normal(mu=[[5.0], [5.0], [5.0], [5.0]], sigma=1.0, columns=["t"])
    metric = ConcordanceHarrell()
    assert metric(y_true, y_pred_const, C_true=c_true) == pytest.approx(0.5)

    # Censored pair (0, 1) has tied scores; comparable pair (2, 3) is concordant -> 1.0
    y_pred_mixed = Normal(mu=[[10.0], [10.0], [1.0], [2.0]], sigma=1.0, columns=["t"])
    assert metric(y_true, y_pred_mixed, C_true=c_true) == pytest.approx(1.0)


def test_charrell_integer_y_true_ties():
    """Integer y_true must not truncate fractional tie_score values in nconc_mat."""
    from skpro.distributions import Normal
    from skpro.metrics.survival._c_harrell import ConcordanceHarrell

    y_true = pd.DataFrame({"t": [1, 2, 3, 4]})
    y_pred = Normal(mu=[[5.0], [5.0], [5.0], [5.0]], sigma=1.0, columns=["t"])
    metric = ConcordanceHarrell()
    assert metric(y_true, y_pred) == pytest.approx(0.5)


def test_charrell_index_normalization_single_comparable_pair():
    """Index normalization keeps exact scores when a sample has 1 comparable pair."""
    from skpro.distributions import Normal
    from skpro.metrics.survival._c_harrell import ConcordanceHarrell

    # Two uncensored samples: each has exactly 1 comparable pair
    y_true = pd.DataFrame({"t": [1.0, 2.0]})
    y_pred_conc = Normal(mu=[[1.0], [2.0]], sigma=0.1, columns=["t"])
    y_pred_disc = Normal(mu=[[2.0], [1.0]], sigma=0.1, columns=["t"])
    metric = ConcordanceHarrell(normalization="index")

    assert metric(y_true, y_pred_conc) == pytest.approx(1.0)
    assert metric(y_true, y_pred_disc) == pytest.approx(0.0)

    # Sample 0 has 0 comparable pairs (censored earliest), samples 1 and 2 have 1 each
    y_true_cens = pd.DataFrame({"t": [1.0, 2.0, 3.0]})
    c_true_cens = pd.DataFrame({"t": [1, 0, 1]})
    y_pred_cens = Normal(mu=[[1.0], [2.0], [3.0]], sigma=0.1, columns=["t"])
    by_idx = metric.evaluate_by_index(y_true_cens, y_pred_cens, C_true=c_true_cens)
    assert by_idx["t"].tolist() == pytest.approx([0.5, 1.0, 1.0])
