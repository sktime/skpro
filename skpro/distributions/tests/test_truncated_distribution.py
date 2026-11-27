"""Tests for TruncatedDistribution validation."""

import pytest

from skpro.distributions import Normal, TruncatedDistribution


def test_truncated_distribution_rejects_inverted_bounds():
    base = Normal(mu=0, sigma=1)
    with pytest.raises(ValueError, match="lower bound"):
        TruncatedDistribution(base, lower=2, upper=1)


def test_truncated_distribution_rejects_empty_mass():
    base = Normal(mu=0, sigma=1)
    with pytest.raises(ValueError, match="no probability mass"):
        TruncatedDistribution(base, lower=10, upper=11)