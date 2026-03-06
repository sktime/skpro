"""Tests for the Fisk (log-logistic) probability distribution."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pytest

from skpro.distributions.fisk import Fisk
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize("params", Fisk.get_test_params())
def test_fisk_basic(params):
    """Test basic construction and method availability of Fisk distribution."""
    d = Fisk(**params)

    # Check that distribution is constructed
    assert d is not None


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_fisk_mean_var():
    """Test that mean and variance match the closed-form formulas."""
    alpha, beta = 2.0, 4.0  # beta > 2 so both mean and var are defined

    d = Fisk(alpha=alpha, beta=beta)

    # expected mean = alpha * (pi/beta) / sin(pi/beta)
    pb = np.pi / beta
    expected_mean = alpha * pb / np.sin(pb)

    # expected var = alpha^2 * (2pi/beta / sin(2pi/beta) - (pi/beta)^2/sin^2(pi/beta))
    two_pb = 2 * np.pi / beta
    expected_var = alpha**2 * (two_pb / np.sin(two_pb) - (pb / np.sin(pb)) ** 2)

    mean_val = d.mean().values.flat[0]
    var_val = d.var().values.flat[0]

    np.testing.assert_allclose(mean_val, expected_mean, rtol=1e-6)
    np.testing.assert_allclose(var_val, expected_var, rtol=1e-6)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_fisk_pdf_cdf_ppf():
    """Test pdf, cdf and ppf for known values; verify cdf(ppf(p)) ~ p."""
    alpha, beta = 1.0, 3.0
    d = Fisk(alpha=alpha, beta=beta)

    # At x = alpha, CDF should be 0.5 (median = alpha for any beta)
    cdf_at_alpha = d.cdf(alpha)
    np.testing.assert_allclose(
        np.asarray(cdf_at_alpha).flat[0], 0.5, atol=1e-8
    )

    # ppf(0.5) should return alpha (the median)
    ppf_at_half = d.ppf(0.5)
    np.testing.assert_allclose(
        np.asarray(ppf_at_half).flat[0], alpha, atol=1e-8
    )

    # PDF > 0 for positive x
    pdf_val = d.pdf(alpha)
    assert np.asarray(pdf_val).flat[0] > 0

    # PDF = 0 for x <= 0
    pdf_neg = d.pdf(0.0)
    np.testing.assert_allclose(np.asarray(pdf_neg).flat[0], 0.0, atol=1e-10)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_fisk_log_pdf():
    """Test that log_pdf is the log of pdf for positive x."""
    alpha, beta = 2.0, 3.0
    d = Fisk(alpha=alpha, beta=beta)

    x = 1.5
    log_pdf_val = np.asarray(d.log_pdf(x)).flat[0]
    pdf_val = np.asarray(d.pdf(x)).flat[0]

    np.testing.assert_allclose(log_pdf_val, np.log(pdf_val), rtol=1e-6)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_fisk_cdf_ppf_roundtrip():
    """Test that ppf(cdf(x)) ≈ x (round-trip consistency)."""
    alpha, beta = 2.0, 4.0
    d = Fisk(alpha=alpha, beta=beta)

    x_vals = np.array([0.5, 1.0, 2.0, 5.0])
    for x in x_vals:
        p = np.asarray(d.cdf(x)).flat[0]
        x_back = np.asarray(d.ppf(p)).flat[0]
        np.testing.assert_allclose(x_back, x, rtol=1e-6, err_msg=f"Failed at x={x}")
