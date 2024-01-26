"""Tests for quantile-parameterized distributions."""

import pytest

from skpro.distributions.qpd import QPD_B, QPD_S, QPD_U
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(QPD_B),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_qpd_b_simple_use():
    """Test simple use of qpd with bounded mode."""
    from skpro.distributions.qpd import QPD_B

    qpd = QPD_B(
        alpha=0.2,
        qv_low=[1, 2],
        qv_median=[3, 4],
        qv_high=[5, 6],
        lower=0,
        upper=10,
    )

    qpd.mean()


@pytest.mark.skipif(
    not run_test_for_class(QPD_S),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_qpd_s_simple_use():
    """Test simple use of qpd with semi-bounded mode."""
    from skpro.distributions.qpd import QPD_S

    qpd = QPD_S(
        alpha=0.2,
        qv_low=[1, 2],
        qv_median=[3, 4],
        qv_high=[5, 6],
        lower=0,
    )

    qpd.mean()


@pytest.mark.skipif(
    not run_test_for_class(QPD_U),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_qpd_u_simple_use():
    """Test simple use of qpd with unbounded mode."""
    from skpro.distributions.qpd import QPD_U

    qpd = QPD_U(
        alpha=0.2,
        qv_low=[1, 2],
        qv_median=[3, 4],
        qv_high=[5, 6],
    )

    qpd.mean()
