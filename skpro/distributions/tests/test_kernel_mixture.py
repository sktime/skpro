"""Tests for KernelMixture distribution."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["amaydixit11"]

import numpy as np
import pandas as pd
import pytest

# np.trapezoid was added in NumPy 2.0; np.trapz removed in NumPy 2.4
try:
    _trapezoid = np.trapezoid
except AttributeError:
    _trapezoid = np.trapz
from skbase.utils.dependencies import _check_soft_dependencies

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
        integral = _trapezoid(pdfs, xs)
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

    def test_cdf_pdf_consistency(self, simple_km):
        """Test that numerical derivative of CDF ~ PDF."""
        xs = np.linspace(-2, 6, 50)
        eps = 1e-5
        for x in xs:
            pdf_val = simple_km.pdf(x)
            cdf_deriv = (simple_km.cdf(x + eps) - simple_km.cdf(x - eps)) / (2 * eps)
            assert abs(pdf_val - cdf_deriv) < 1e-3

    def test_sample_mean_convergence(self, simple_km):
        """Test that sample mean converges to analytical mean."""

        samples = simple_km.sample(10000)
        sample_mean = samples.values.mean()
        assert abs(sample_mean - simple_km.mean()) < 0.1

    def test_random_state_reproducibility(self):
        """Test that random_state produces reproducible samples."""
        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        km1 = KernelMixture(
            support=support, bandwidth=0.5, kernel="gaussian", random_state=42
        )
        km2 = KernelMixture(
            support=support, bandwidth=0.5, kernel="gaussian", random_state=42
        )
        s1 = km1.sample(100)
        s2 = km2.sample(100)
        np.testing.assert_array_equal(s1.values, s2.values)

    def test_auto_bandwidth_scott(self):
        """Test Scott bandwidth rule."""
        support = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        km = KernelMixture(support=support, bandwidth="scott", kernel="gaussian")
        expected = len(support) ** (-1.0 / 5.0) * np.std(support, ddof=1)
        assert abs(km._bandwidth - expected) < 1e-10

    def test_auto_bandwidth_silverman(self):
        """Test Silverman bandwidth rule."""
        support = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        km = KernelMixture(support=support, bandwidth="silverman", kernel="gaussian")
        expected = (4.0 / (3.0 * 5)) ** (1.0 / 5.0) * np.std(support, ddof=1)
        assert abs(km._bandwidth - expected) < 1e-10

    def test_subsetting_2d(self):
        """Test that iloc subsetting works correctly."""
        km = KernelMixture(
            support=[0.0, 1.0, 2.0], bandwidth=0.5, kernel="gaussian",
            index=pd.RangeIndex(3), columns=pd.Index(["a", "b"]),
        )
        sub = km.iloc[[0, 1], [0]]
        assert sub.shape == (2, 1)
        sub_scalar = km.iloc[0, 0]
        assert sub_scalar.shape == ()

    @pytest.mark.parametrize("rule", ["scott", "silverman"])
    def test_auto_bandwidth_single_element(self, rule):
        """Test that string bandwidth with single-element support doesn't produce nan."""
        km = KernelMixture(support=[5.0], bandwidth=rule, kernel="gaussian")
        assert np.isfinite(km._bandwidth)
        assert km._bandwidth > 0

    def test_invalid_kernel_type_raises(self):
        """Test that non-string non-distribution kernel raises TypeError."""
        with pytest.raises(TypeError, match="kernel must be a string"):
            KernelMixture(support=[0, 1], bandwidth=1.0, kernel=42)

    def test_distribution_kernel_pdf(self):
        """Test that Normal(0,1) kernel gives same result as gaussian."""
        from skpro.distributions.normal import Normal
        support = np.array([0.0, 1.0, 2.0])
        bw = 0.5
        km_str = KernelMixture(support=support, bandwidth=bw, kernel="gaussian")
        km_dist = KernelMixture(
            support=support, bandwidth=bw, kernel=Normal(mu=0, sigma=1)
        )
        for x in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]:
            assert abs(km_str.pdf(x) - km_dist.pdf(x)) < 1e-6

    def test_distribution_kernel_cdf(self):
        """Test that Normal(0,1) kernel CDF matches gaussian CDF."""
        from skpro.distributions.normal import Normal
        support = np.array([0.0, 1.0, 2.0])
        bw = 0.5
        km_str = KernelMixture(support=support, bandwidth=bw, kernel="gaussian")
        km_dist = KernelMixture(
            support=support, bandwidth=bw, kernel=Normal(mu=0, sigma=1)
        )
        for x in [-1.0, 0.0, 1.0, 2.0, 3.0]:
            assert abs(km_str.cdf(x) - km_dist.cdf(x)) < 1e-6

    def test_distribution_kernel_mean_var(self):
        """Test that Normal(0,1) kernel gives same mean/var as gaussian."""
        from skpro.distributions.normal import Normal
        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        bw = 0.5
        km_str = KernelMixture(support=support, bandwidth=bw, kernel="gaussian")
        km_dist = KernelMixture(
            support=support, bandwidth=bw, kernel=Normal(mu=0, sigma=1)
        )
        assert abs(km_str.mean() - km_dist.mean()) < 1e-10
        assert abs(km_str.var() - km_dist.var()) < 1e-10

    def test_distribution_kernel_sample(self):
        """Test that sampling with a distribution kernel works."""
        from skpro.distributions.normal import Normal
        km = KernelMixture(
            support=[0.0, 1.0, 2.0], bandwidth=0.5,
            kernel=Normal(mu=0, sigma=1),
        )
        assert np.isfinite(km.sample())
        s_multi = km.sample(10)
        assert isinstance(s_multi, pd.DataFrame)
        assert s_multi.shape[0] == 10

    @pytest.mark.skipif(
        not _check_soft_dependencies("sklearn", severity="none"),
        reason="sklearn not available",
    )
    def test_sklearn_parity(self):
        """Test parity with sklearn KernelDensity for Gaussian kernel."""
        from sklearn.neighbors import KernelDensity
        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        bw = 0.5
        km = KernelMixture(support=support, bandwidth=bw, kernel="gaussian")
        kde = KernelDensity(bandwidth=bw, kernel="gaussian")
        kde.fit(support.reshape(-1, 1))
        xs = np.linspace(-2, 6, 50)
        for x in xs:
            skpro_pdf = km.pdf(x)
            sklearn_pdf = np.exp(kde.score_samples(np.array([[x]]))[0])
            assert abs(skpro_pdf - sklearn_pdf) < 1e-6
