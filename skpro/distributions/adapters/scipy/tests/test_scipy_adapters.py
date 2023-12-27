"""Tests for adapters for probability distribution objects, scipy facing."""

import numpy as np
import pandas as pd


def test_empirical_from_discrete():
    """Test empirical_from_discrete."""
    from scipy.stats import rv_discrete

    from skpro.distributions.adapters.scipy._empirical import empirical_from_discrete

    xk = np.arange(7)
    pk = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
    pk2 = (0.1, 0.1, 0.4, 0.0, 0.1, 0.2, 0.1)

    dist1 = rv_discrete(name="custm", values=(xk, pk))
    dist2 = rv_discrete(name="custm", values=(xk, pk2))

    emp = empirical_from_discrete([dist1, dist2])
    assert isinstance(emp.spl, pd.DataFrame)
    assert isinstance(emp.weights, pd.Series)
    assert emp.spl.shape == (14, 1)
    assert emp.weights.shape == (14,)
    expected_idx = pd.MultiIndex.from_arrays(
        [[0, 1, 2, 3, 4, 5, 6] * 2, [0] * 7 + [1] * 7]
    )
    assert np.all(emp.spl.index == expected_idx)
    assert np.all(emp.spl.columns == [0])

    emp2 = empirical_from_discrete(
        [dist1, dist2], index=pd.Index(["foo", "bar"]), columns=["abc"]
    )
    assert isinstance(emp2.spl, pd.DataFrame)
    assert isinstance(emp2.weights, pd.Series)
    assert emp2.spl.shape == (14, 1)
    assert emp2.weights.shape == (14,)
    expected_idx = pd.MultiIndex.from_arrays(
        [[0, 1, 2, 3, 4, 5, 6] * 2, ["foo"] * 7 + ["bar"] * 7]
    )
    assert np.all(emp2.spl.index == expected_idx)
    assert np.all(emp2.spl.columns == ["abc"])
