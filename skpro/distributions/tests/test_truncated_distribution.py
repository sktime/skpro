"""Tests for TruncatedDistribution validation."""

import numpy as np
import pandas as pd
import pytest

from skpro.distributions import Normal, TruncatedDistribution


def test_truncated_distribution_rejects_inverted_bounds():
    base = Normal(mu=0, sigma=1)
    with pytest.raises(ValueError, match="lower bound"):
        TruncatedDistribution(base, lower=2, upper=1)


def test_truncated_distribution_rejects_inverted_bounds_array():
    idx = pd.Index([0, 1])
    cols = pd.Index(["a"])
    base = Normal(mu=[[0.0], [0.0]], sigma=1.0, index=idx, columns=cols)
    lower = np.array([[2.0], [0.0]])
    upper = np.array([[1.0], [1.0]])
    with pytest.raises(ValueError, match="lower bound"):
        TruncatedDistribution(base, lower=lower, upper=upper)


def test_truncated_distribution_rejects_empty_mass():
    base = Normal(mu=0, sigma=1)
    with pytest.raises(ValueError, match="no probability mass"):
        TruncatedDistribution(base, lower=10, upper=11)
