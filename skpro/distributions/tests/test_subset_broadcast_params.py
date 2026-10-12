"""Subsetting distributions whose parameters are broadcast over rows or columns.

A parameter of shape ``(n_rows, 1)`` or ``(1, n_cols)`` is accepted by the
constructor and broadcast against the other parameters, so ``iloc`` and ``iat``
have to repeat it for the selected rows and columns, not index it by position.
"""

import numpy as np
import pandas as pd
import pytest

from skpro.distributions import Normal, Poisson

INDEX = pd.Index(["r0", "r1"])
COLUMNS = pd.Index(["a", "b", "c"])
MU = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])

DISTRIBUTIONS = {
    "Normal, sigma per row": lambda: Normal(
        mu=MU, sigma=np.array([[1.0], [2.0]]), index=INDEX, columns=COLUMNS
    ),
    "Normal, sigma per column": lambda: Normal(
        mu=MU, sigma=np.array([[1.0, 2.0, 3.0]]), index=INDEX, columns=COLUMNS
    ),
    "Normal, mu per row and sigma per column": lambda: Normal(
        mu=np.array([[0.0], [3.0]]),
        sigma=np.array([[1.0, 2.0, 3.0]]),
        index=INDEX,
        columns=COLUMNS,
    ),
    "Normal, mu in one row": lambda: Normal(
        mu=np.array([[0.0, 1.0, 2.0]]),
        sigma=np.array([[1.0], [2.0]]),
        index=INDEX,
        columns=COLUMNS,
    ),
    "Poisson, mu per row": lambda: Poisson(
        mu=np.array([[1.0], [2.0]]), index=INDEX, columns=COLUMNS
    ),
}

# selection on the distribution, and the equivalent row and column positions
ALL = [0, 1, 2]
SELECTIONS = {
    "row": (lambda d: d.iloc[[1]], [1], ALL),
    "column": (lambda d: d.iloc[:, [2]], [0, 1], [2]),
    "rows and columns": (lambda d: d.iloc[[1, 0], [2, 0]], [1, 0], [2, 0]),
    "cell by list iloc": (lambda d: d.iloc[[1], [2]], [1], [2]),
    "cell by iat": (lambda d: d.iat[1, 2], [1], [2]),
    "cell by scalar iloc": (lambda d: d.iloc[1, 2], [1], [2]),
    "scalar row": (lambda d: d.iloc[0], [0], ALL),
    "scalar column": (lambda d: d.iloc[:, 1], [0, 1], [1]),
}


@pytest.mark.parametrize("make", DISTRIBUTIONS.values(), ids=DISTRIBUTIONS.keys())
@pytest.mark.parametrize(
    "select, rows, cols", SELECTIONS.values(), ids=SELECTIONS.keys()
)
def test_subsetting_keeps_broadcast_parameters_of_the_selected_cells(
    make, select, rows, cols
):
    distr = make()
    # a different point per cell, so that using the wrong cell's parameters shows up
    x = pd.DataFrame(np.arange(6.0).reshape(2, 3), index=INDEX, columns=COLUMNS)
    expected = distr.cdf(x).iloc[rows, cols]
    x_selected = x.iloc[rows, cols]

    sub = select(distr)

    if sub.ndim == 0:
        got = sub.cdf(float(x_selected.iloc[0, 0]))
    else:
        assert sub.shape == x_selected.shape
        got = sub.cdf(x_selected)

    np.testing.assert_allclose(
        np.asarray(got, dtype=float).reshape(-1),
        expected.to_numpy(dtype=float).reshape(-1),
    )


@pytest.mark.parametrize("make", DISTRIBUTIONS.values(), ids=DISTRIBUTIONS.keys())
def test_subsetting_to_no_rows_or_no_columns_keeps_the_shape(make):
    distr = make()

    assert distr.iloc[[]].shape == (0, 3)
    assert distr.iloc[:, []].shape == (2, 0)


@pytest.mark.parametrize("make", DISTRIBUTIONS.values(), ids=DISTRIBUTIONS.keys())
def test_repeating_a_row_repeats_its_broadcast_parameters(make):
    distr = make()

    sub = distr.iloc[[1, 1, 0]]

    assert sub.shape == (3, 3)
    for stat in ("mean", "var"):
        np.testing.assert_allclose(
            getattr(sub, stat)().to_numpy(),
            getattr(distr, stat)().iloc[[1, 1, 0]].to_numpy(),
        )
