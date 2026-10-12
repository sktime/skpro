"""Scalar positional indexers on distributions that subset their own parameters.

``dist.iloc[0]``, ``dist.iloc[:, 1]`` and ``dist.iloc[0, 1]`` must select the same cells
as the equivalent list indexers, including the part of ``p`` that belongs to them.
"""

import numpy as np
import pandas as pd
import pytest

from skpro.distributions import Hurdle, NegativeBinomial, ZeroInflated

INDEX = pd.Index(["r0", "r1"])
COLUMNS = pd.Index(["a", "b", "c"])

P_VALUES = {
    "scalar p": 0.3,
    "p per row": np.array([0.2, 0.3]).reshape(-1, 1),
    "p per cell": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
}

# (scalar indexer, equivalent list indexer) pairs, as positions into a 2 x 3 frame
SELECTIONS = {
    "row": ((0, slice(None)), ([0], [0, 1, 2])),
    "last row": ((-1, slice(None)), ([-1], [0, 1, 2])),
    "column": ((slice(None), 1), ([0, 1], [1])),
    "last column": ((slice(None), -1), ([0, 1], [-1])),
    "cell": ((1, 2), ([1], [2])),
    "last cell": ((-1, -1), ([-1], [-1])),
    "scalar row, list columns": ((1, [0, 2]), ([1], [0, 2])),
    "list rows, scalar column": (([0, 1], 2), ([0, 1], [2])),
}


def _negative_binomial():
    mu = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return NegativeBinomial(mu=mu, alpha=1.0, index=INDEX, columns=COLUMNS)


@pytest.fixture(params=[ZeroInflated, Hurdle])
def distr_class(request):
    return request.param


CASES = [
    pytest.param(name, ix, p_name, id=f"{name}-{p_name}")
    for name, ix in SELECTIONS.items()
    for p_name in P_VALUES
    # a single cell goes through ``iat``, which fails on a per-row ``p``, see #1202
    if not (p_name == "p per row" and "cell" in name and "list" not in name)
]


@pytest.mark.parametrize("name, ix, p_name", CASES)
def test_scalar_iloc_selects_the_same_cells_as_list_iloc(distr_class, name, ix, p_name):
    scalar_ix, list_ix = ix
    distr = distr_class(p=P_VALUES[p_name], distribution=_negative_binomial())
    # a different point per cell, so a wrong cell or a wrong p shows up in the cdf
    x = pd.DataFrame(np.arange(6.0).reshape(2, 3), index=INDEX, columns=COLUMNS)
    expected = distr.cdf(x).iloc[list_ix[0], list_ix[1]]
    x_selected = x.iloc[list_ix[0], list_ix[1]]

    sub = distr.iloc[scalar_ix[0], scalar_ix[1]]

    if sub.ndim == 0:
        got = sub.cdf(float(x_selected.iloc[0, 0]))
    else:
        assert sub.shape == x_selected.shape
        assert list(sub.index) == list(x_selected.index)
        assert list(sub.columns) == list(x_selected.columns)
        got = sub.cdf(x_selected)

    np.testing.assert_allclose(
        np.asarray(got, dtype=float).reshape(-1),
        expected.to_numpy(dtype=float).reshape(-1),
    )


def test_iat_and_scalar_iloc_agree(distr_class):
    distr = distr_class(p=P_VALUES["p per cell"], distribution=_negative_binomial())

    from_iat = distr.iat[1, 2]
    from_iloc = distr.iloc[1, 2]

    assert from_iat.ndim == from_iloc.ndim == 0
    for point in (0.0, 2.0, 5.0):
        assert from_iat.cdf(point) == pytest.approx(from_iloc.cdf(point))
