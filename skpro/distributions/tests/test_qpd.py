"""Tests for quantile-parameterized distributions."""

import numpy as np
import pytest

from skpro.distributions.qpd import QPD_B, QPD_S, QPD_U
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(QPD_B),
    reason="run test only if softdeps are present and incrementally (if requested)",  #
)
def test_qpd_b_simple_use():
    """Test simple use of qpd with bounded mode."""
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
    not run_test_for_class(QPD_B),
    reason="run test only if softdeps are present and incrementally (if requested)",  #
)
def test_qpd_b_pdf():
    """Test pdf of qpd with bounded mode."""
    # these parameters should produce a uniform on -0.5, 0.5
    qpd_linear = QPD_B(
        alpha=0.2,
        qv_low=-0.3,
        qv_median=0,
        qv_high=0.3,
        lower=-0.5,
        upper=0.5,
    )
    x = np.linspace(-0.45, 0.45, 100)
    pdf_vals = [qpd_linear.pdf(x_) for x_ in x]
    np.testing.assert_allclose(pdf_vals, 1.0, rtol=1e-5)


@pytest.mark.skipif(
    not run_test_for_class(QPD_S),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_qpd_s_simple_use():
    """Test simple use of qpd with semi-bounded mode."""
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
    """Test simple use of qpd with un-bounded mode."""
    qpd = QPD_U(
        alpha=0.2,
        qv_low=[1, 2],
        qv_median=[3, 4],
        qv_high=[5, 6],
    )

    qpd.mean()


@pytest.mark.skipif(
    not run_test_for_class(QPD_U),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_qpd_u_pdf_is_cdf_derivative():
    """Test that the pdf of qpd with un-bounded mode is the derivative of the cdf."""
    from scipy.integrate import quad

    qpd = QPD_U(alpha=0.1, qv_low=1.0, qv_median=3.0, qv_high=7.0)

    x = np.linspace(-5, 20, 11)
    h = 1e-6
    cdf_der = [(qpd.cdf(x_ + h) - qpd.cdf(x_ - h)) / (2 * h) for x_ in x]
    pdf_vals = [qpd.pdf(x_) for x_ in x]
    np.testing.assert_allclose(pdf_vals, cdf_der, rtol=1e-5, atol=1e-8)

    integral = quad(lambda x_: float(qpd.pdf(x_)), -np.inf, np.inf)[0]
    np.testing.assert_allclose(integral, 1.0, rtol=1e-6)


@pytest.mark.skipif(
    not run_test_for_class(QPD_U),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_qpd_u_symmetric():
    """Test qpd with un-bounded mode for a symmetric quantile triplet.

    The quantile values must be recovered by ppf, and the distribution must be
    the limit of nearly symmetric triplets, which is the base distribution
    with location ``qv_median`` and scale ``(qv_high - qv_median) / phi.ppf(1-alpha)``.
    """
    from scipy.stats import norm

    alpha = 0.2
    qpd = QPD_U(alpha=alpha, qv_low=-0.3, qv_median=0.0, qv_high=0.3)

    np.testing.assert_allclose(
        [qpd.ppf(p) for p in [alpha, 0.5, 1 - alpha]], [-0.3, 0.0, 0.3], atol=1e-12
    )

    expected = norm(loc=0.0, scale=0.3 / norm.ppf(1 - alpha))
    x = np.linspace(-1, 1, 9)
    np.testing.assert_allclose([qpd.cdf(x_) for x_ in x], expected.cdf(x))
    np.testing.assert_allclose([qpd.pdf(x_) for x_ in x], expected.pdf(x))

    qpd_near = QPD_U(alpha=alpha, qv_low=-0.3, qv_median=0.0, qv_high=0.3 + 1e-7)
    ps = [0.01, 0.3, 0.7, 0.99]
    np.testing.assert_allclose(
        [qpd.ppf(p) for p in ps], [qpd_near.ppf(p) for p in ps], atol=1e-5
    )
