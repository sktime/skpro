"""Tests for Harell's C-index."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import pandas as pd
import pytest


@pytest.mark.parametrize("concordant", [True, False])
@pytest.mark.parametrize("pass_c", [True, False])
@pytest.mark.parametrize("normalization", ["overall", "index"])
def test_charrell_logic(concordant, pass_c, normalization):
    """Test the logic of the Harrell's C-index metric.

    Parameters
    ----------
    concordant : bool, optional, default=True
        If True, the test examples are fully concordant.
        If False, the test examples are fully discordant.
    pass_c : bool, optional, default=True
        If True, the `c_true` argument is passed to the metric.
        If False, the `c_true` argument is not passed to the metric.
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
    metric = ConcordanceHarrell(normalization=normalization)
    metric_args = {"y_true": y_true, "y_pred": y_pred}
    if pass_c:
        metric_args["c_true"] = c_true

    res = metric(**metric_args)
    res_by_index = metric.evaluate_by_index(**metric_args)

    # test assumptions
    # if concordant, the result should be 1
    # if discordant, the result should be 0
    assert res == concordant
    assert res_by_index.shape == y_true.shape
    assert (res_by_index == concordant).all().all()
