"""Tests for KernelMixture distribution."""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["amaydixit11"]

import numpy as np
import pandas as pd
import pytest

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
            support=support,
            bandwidth=0.5,
            kernel="gaussian",
            weights=weights,
        )

    @pytest.mark.parametrize(
        "kernel", ["gaussian", "epanechnikov", "tophat", "cosine", "linear"]
    )
    def test_pdf_integrates_to_one(self, kernel):
        """Test pdf integrates to 1."""
        support = np.array([0.0, 1.0, 2.0, 3.0])
        km = KernelMixture(support=support, bandwidth=0.5, kernel=kernel)
        xs = np.linspace(-5, 8, 10000)
        pdfs = np.array([km.pdf(x) for x in xs])
        integral = _trapezoid(pdfs, xs)
        assert abs(integral - 1.0) < 0.01

    def test_mean_correctness(self, simple_km):
        """Test mean equals weighted average of support."""
        assert abs(simple_km.mean() - 2.0) < 1e-10

    def test_weighted_mean(self, weighted_km):
        """Test weighted mixture mean."""
        assert abs(weighted_km.mean()) < 1e-10

    def test_var_correctness(self, simple_km):
        """Test variance formula."""
        h = 0.5
        support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        expected_var = h**2 * 1.0 + np.var(support)
        assert abs(simple_km.var() - expected_var) < 1e-10

    def test_cdf_monotonicity(self, simple_km):
        """Test CDF is non-decreasing."""
        xs = np.linspace(-3, 7, 200)
        cdfs = np.array([simple_km.cdf(x) for x in xs])
        assert np.all(np.diff(cdfs) >= -1e-10)

    def test_cdf_limits(self, simple_km):
        """Test CDF limits."""
        assert simple_km.cdf(-20.0) < 1e-6
        assert simple_km.cdf(25.0) > 1 - 1e-6

    def test_sample_shape_scalar(self, simple_km):
        """Test scalar sample shape."""
        s = simple_km.sample()
        assert np.isscalar(s)
        s_multi = simple_km.sample(10)
        assert isinstance(s_multi, pd.DataFrame)
        assert s_multi.shape[0] == 10

    def test_sample_shape_2d(self):
        """Test 2D sample shape."""
        km = KernelMixture(
            support=[0.0, 1.0, 2.0],
            bandwidth=0.5,
            kernel="gaussian",
            index=pd.RangeIndex(3),
            columns=pd.Index(["a", "b"]),
        )
        s = km.sample()
        assert s.shape == (3, 2)
        s_multi = km.sample(5)
        assert s_multi.shape == (15, 2)

    def test_invalid_kernel_raises(self):
        """Test invalid kernel raises."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            KernelMixture(support=[0, 1], bandwidth=1.0, kernel="invalid")

    def test_weight_length_mismatch_raises(self):
        """Test mismatched weights raises."""
        with pytest.raises(ValueError, match="weights length"):
            KernelMixture(support=[0, 1, 2], bandwidth=1.0, weights=[0.5, 0.5])

    def test_non_positive_bandwidth_raises(self):
        """Test non-positive bandwidth raises."""
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            KernelMixture(support=[0, 1, 2], bandwidth=0.0)
        with pytest.raises(ValueError, match="bandwidth must be positive"):
            KernelMixture(support=[0, 1, 2], bandwidth=-1.0)

    def test_negative_weights_raises(self):
        """Test negative weights raise."""
        with pytest.raises(ValueError, match="non-negative"):
            KernelMixture(support=[0, 1, 2], bandwidth=1.0, weights=[1.0, -0.5, 0.5])

    def test_log_pdf_consistency(self, simple_km):
        """Test log_pdf consistency with pdf."""
        for x in [0.0, 1.0, 2.0, 3.0]:
            assert abs(simple_km.log_pdf(x) - np.log(simple_km.pdf(x))) < 1e-10

    @pytest.mark.parametrize(
        "kernel", ["gaussian", "epanechnikov", "tophat", "cosine", "linear"]
    )
    def test_all_kernels_basic(self, kernel):
        """Test basic functionality for all kernels."""
        km = KernelMixture(support=[0.0, 1.0, 2.0], bandwidth=0.5, kernel=kernel)
        assert np.isfinite(km.mean())
        assert km.var() > 0
        assert km.pdf(1.0) > 0
        assert 0 <= km.cdf(1.0) <= 1
        assert np.isfinite(km.sample())

    def test_cdf_pdf_consistency(self, simple_km):
        """Test CDF derivative matches PDF."""
        xs = np.linspace(-2, 6, 50)
        eps = 1e-5
        for x in xs:
            pdf_val = simple_km.pdf(x)
            cdf_deriv = (simple_km.cdf(x + eps) - simple_km.cdf(x - eps)) / (2 * eps)
            assert abs(pdf_val - cdf_deriv) < 1e-3

    def test_sample_mean_convergence(self, simple_km):
        """Test sample mean convergence."""
        samples = simple_km.sample(10000)
        sample_mean = samples.values.mean()
        assert abs(sample_mean - simple_km.mean()) < 0.1

    def test_random_state_reproducibility(self):
        """Test random_state reproducibility."""
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
        """Test iloc subsetting."""
        km = KernelMixture(
            support=[0.0, 1.0, 2.0],
            bandwidth=0.5,
            kernel="gaussian",
            index=pd.RangeIndex(3),
            columns=pd.Index(["a", "b"]),
        )
        sub = km.iloc[[0, 1], [0]]
        assert sub.shape == (2, 1)
        sub_scalar = km.iloc[0, 0]
        assert sub_scalar.shape == ()

    @pytest.mark.parametrize("rule", ["scott", "silverman"])
    def test_auto_bandwidth_single_element(self, rule):
        """Test bandwidth with single support point."""
        km = KernelMixture(support=[5.0], bandwidth=rule, kernel="gaussian")
        assert np.isfinite(km._bandwidth)
        assert km._bandwidth > 0

    def test_invalid_kernel_type_raises(self):
        """Test non-string/distribution kernel raises."""
        with pytest.raises(TypeError, match="kernel must be a string"):
            KernelMixture(support=[0, 1], bandwidth=1.0, kernel=42)

    def test_non_scalar_kernel_raises(self):
        """Test non-scalar distribution kernel raises."""
        from skpro.distributions.normal import Normal

        kernel_2d = Normal(
            mu=[[0, 0]],
            sigma=[[1, 1]],
            index=pd.RangeIndex(1),
            columns=pd.Index(["a", "b"]),
        )
        with pytest.raises(ValueError, match="scalar"):
            KernelMixture(support=[0, 1], bandwidth=1.0, kernel=kernel_2d)

    def test_nonzero_mean_kernel_warns(self):
        """Test non-zero mean kernel warns."""
        from skpro.distributions.normal import Normal

        with pytest.warns(UserWarning, match="non-zero mean"):
            KernelMixture(
                support=[0, 1, 2], bandwidth=0.5, kernel=Normal(mu=5, sigma=1)
            )

    def test_distribution_kernel_rng_warns(self):
        """Test distribution kernel RNG warning."""
        from skpro.distributions.normal import Normal

        km = KernelMixture(
            support=[0, 1, 2],
            bandwidth=0.5,
            kernel=Normal(mu=0, sigma=1),
            random_state=42,
        )
        with pytest.warns(UserWarning, match="random_state"):
            km.sample(10)

    def test_distribution_kernel_pdf(self):
        """Test Normal kernel matches gaussian."""
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
        """Test Normal kernel CDF matches gaussian."""
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
        """Test Normal kernel mean/var matches gaussian."""
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
        """Test distribution kernel sampling."""
        from skpro.distributions.normal import Normal

        km = KernelMixture(
            support=[0.0, 1.0, 2.0],
            bandwidth=0.5,
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
        """Test parity with sklearn KernelDensity."""
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
