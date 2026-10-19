"""Tests for quantile-parameterized distributions."""

import warnings

import numpy as np
import pandas as pd
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
    not run_test_for_class(QPD_S),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_qpd_s_cdf_pdf_outside_support():
    """cdf and pdf of QPD_S are 0 at or below lower, without warnings, see #885."""
    qpd = QPD_S(
        alpha=0.2,
        qv_low=[-0.3],
        qv_median=[0.0],
        qv_high=[0.3],
        lower=-0.5,
        index=pd.RangeIndex(4),
        columns=pd.Index(["a"]),
    )
    x = pd.DataFrame({"a": [-1.0, -0.5, 0.0, 1.0]})

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cdf = qpd.cdf(x).to_numpy().ravel()
        pdf = qpd.pdf(x).to_numpy().ravel()

    np.testing.assert_allclose(cdf[:2], 0.0)
    np.testing.assert_allclose(cdf[2], 0.5)
    assert 0.5 < cdf[3] < 1.0
    np.testing.assert_allclose(pdf[:2], 0.0)
    assert pdf[2] > 0
    assert pdf[3] > 0


@pytest.mark.skipif(
    not run_test_for_class(QPD_B),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_qpd_b_cdf_pdf_outside_support():
    """cdf of QPD_B is 0 below lower and 1 above upper, pdf is 0, no warnings."""
    # these parameters produce a uniform on -0.5, 0.5
    qpd = QPD_B(
        alpha=0.2,
        qv_low=[-0.3],
        qv_median=[0.0],
        qv_high=[0.3],
        lower=-0.5,
        upper=0.5,
        index=pd.RangeIndex(6),
        columns=pd.Index(["a"]),
    )
    x = pd.DataFrame({"a": [-1.0, -0.5, -0.4, 0.4, 0.5, 1.0]})

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        cdf = qpd.cdf(x).to_numpy().ravel()
        pdf = qpd.pdf(x).to_numpy().ravel()

    np.testing.assert_allclose(cdf, [0.0, 0.0, 0.1, 0.9, 1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(pdf, [0.0, 0.0, 1.0, 1.0, 0.0, 0.0], atol=1e-5)
