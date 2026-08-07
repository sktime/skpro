"""Tests for Gumbel distribution."""

import pytest

from skpro.distributions.gumbel import Gumbel
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_gumbel_skew_parameter():
    """Test that Gumbel raises ValueError for invalid skew parameter."""
    with pytest.raises(
        ValueError, match='skew parameter must be either "right" or "left".'
    ):
        Gumbel(mu=0, beta=1, skew="invalid_skew")


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_gumbel_methods():
    """Test basic methods of Gumbel distribution do not raise errors."""
    dist_r = Gumbel(mu=0.0, beta=1.0, skew="right")
    dist_l = Gumbel(mu=0.0, beta=1.0, skew="left")

    # Check that they return values and not errors
    assert dist_r.mean() is not None
    assert dist_r.var() is not None
    assert dist_l.mean() is not None
    assert dist_l.var() is not None

    # check broadcast
    dist_b = Gumbel(mu=[[1, 1], [2, 3], [4, 5]], beta=2, skew="right")
    assert dist_b.shape == (3, 2)
