"""Tests for KernelMixture distribution."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["amaydixit11"]

import numpy as np
import pandas as pd
import pytest

from skpro.distributions.kernel_mixture import KernelMixture
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
class TestKernelMixture:
    """Tests for KernelMixture distribution."""

    @pytest.fixture
    def simple_km(self):
        """Simple Gaussian kernel mixture for testing."""
        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        return KernelMixture(support=support, bandwidth=0.5, kernel="gaussian")

    @pytest.fixture
    def weighted_km(self):
        """Weighted kernel mixture for testing."""
        support = np.array([-1.0, 0.0, 1.0])
        weights = np.array([0.25, 0.5, 0.25])
        return KernelMixture(
            support=support, bandwidth=0.5, kernel="gaussian", weights=weights,
        )

    @pytest.mark.parametrize(
        "kernel", ["gaussian", "epanechnikov", "tophat", "cosine", "linear"]
    )
    def test_pdf_integrates_to_one(self, kernel):
        """Test that pdf integrates to approximately 1."""
        support = np.array([0.0, 1.0, 2.0, 3.0])
        km = KernelMixture(support=support, bandwidth=0.5, kernel=kernel)
        xs = np.linspace(-5, 8, 10000)
        pdfs = np.array([km.pdf(x) for x in xs])
        integral = np.trapezoid(pdfs, xs)
        assert abs(integral - 1.0) < 0.01

    def test_mean_correctness(self, simple_km):
        """Test that mean equals weighted average of support."""
        assert abs(simple_km.mean() - 2.0) < 1e-10

    def test_weighted_mean(self, weighted_km):
        """Test that mean is correct for weighted mixture."""
        assert abs(weighted_km.mean()) < 1e-10

    def test_var_correctness(self, simple_km):
        """Test variance = h^2 * Var(K) + Var_w(support)."""
        h = 0.5
        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        expected_var = h**2 * 1.0 + np.var(support)
        assert abs(simple_km.var() - expected_var) < 1e-10

    def test_cdf_monotonicity(self, simple_km):
        """Test that CDF is non-decreasing."""
        xs = np.linspace(-3, 7, 200)
        cdfs = np.array([simple_km.cdf(x) for x in xs])
        assert np.all(np.diff(cdfs) >= -1e-10)

    def test_cdf_limits(self, simple_km):
        """Test CDF approaches 0 at -inf and 1 at +inf."""
        assert simple_km.cdf(-20.0) < 1e-6
        assert simple_km.cdf(25.0) > 1 - 1e-6

    def test_sample_shape_scalar(self, simple_km):
        """Test sample shape for scalar distribution."""
        s = simple_km.sample()
        assert np.isscalar(s)
        s_multi = simple_km.sample(10)
        assert isinstance(s_multi, pd.DataFrame)
        assert s_multi.shape[0] == 10

    def test_sample_shape_2d(self):
        """Test sample shape for 2D distribution."""
        km = KernelMixture(
            support=[0.0, 1.0, 2.0], bandwidth=0.5, kernel="gaussian",
            index=pd.RangeIndex(3), columns=pd.Index(["a", "b"]),
        )
        s = km.sample()
        assert s.shape == (3, 2)
        s_multi = km.sample(5)
        assert s_multi.shape == (15, 2)

    def test_invalid_kernel_raises(self):
        """Test that invalid kernel raises ValueError."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            KernelMixture(support=[0, 1], bandwidth=1.0, kernel="invalid")

    def test_weight_length_mismatch_raises(self):
        """Test that mismatched weights length raises ValueError."""
        with pytest.raises(ValueError, match="weights length"):
            KernelMixture(support=[0, 1, 2], bandwidth=1.0, weights=[0.5, 0.5])

    def test_log_pdf_consistency(self, simple_km):
        """Test that log_pdf == log(pdf)."""
        for x in [0.0, 1.0, 2.0, 3.0]:
            assert abs(simple_km.log_pdf(x) - np.log(simple_km.pdf(x))) < 1e-10

    @pytest.mark.parametrize(
        "kernel", ["gaussian", "epanechnikov", "tophat", "cosine", "linear"]
    )
    def test_all_kernels_basic(self, kernel):
        """Test basic functionality for all kernel types."""
        km = KernelMixture(support=[0.0, 1.0, 2.0], bandwidth=0.5, kernel=kernel)
        assert np.isfinite(km.mean())
        assert km.var() > 0
        assert km.pdf(1.0) > 0
        assert 0 <= km.cdf(1.0) <= 1
        assert np.isfinite(km.sample())
