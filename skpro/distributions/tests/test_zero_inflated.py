# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for Zero-Inflated distributions."""

import numpy as np
import pytest

from skpro.distributions.zero_inflated import ZeroInflated
from skpro.distributions.zi_negative_binomial import ZINB
from skpro.distributions.zi_poisson import ZIPoisson
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
class TestZIPoisson:
    """Tests for ZIPoisson distribution."""

    @pytest.mark.parametrize("params", ZIPoisson.get_test_params())
    def test_zipoisson_less_than_zero(self, params):
        """Test that negative values return correct probabilities."""
        distribution = ZIPoisson(**params)
        v = -1.0

        funcs_and_expected = [
            (distribution.cdf, 0.0),
            (distribution.pmf, 0.0),
            (distribution.log_pmf, -np.inf),
        ]

        for func, expected in funcs_and_expected:
            values = func(v)
            assert (np.asarray(values) == expected).all()

    @pytest.mark.parametrize("params", ZIPoisson.get_test_params())
    def test_zipoisson_mean_positive(self, params):
        """Test that mean is non-negative."""
        distribution = ZIPoisson(**params)
        mean = distribution.mean()
        assert (np.asarray(mean) >= 0).all()

    @pytest.mark.parametrize("params", ZIPoisson.get_test_params())
    def test_zipoisson_var_positive(self, params):
        """Test that variance is non-negative."""
        distribution = ZIPoisson(**params)
        var = distribution.var()
        assert (np.asarray(var) >= 0).all()

    @pytest.mark.parametrize("params", ZIPoisson.get_test_params())
    def test_zipoisson_cdf_at_zero(self, params):
        """Test CDF at zero includes zero-inflation."""
        distribution = ZIPoisson(**params)
        cdf_at_0 = distribution.cdf(0.0)
        # CDF at 0 should be >= pi (the zero-inflation probability)
        assert (np.asarray(cdf_at_0) >= params["pi"]).all()


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
class TestZINB:
    """Tests for ZINB distribution."""

    @pytest.mark.parametrize("params", ZINB.get_test_params())
    def test_zinb_less_than_zero(self, params):
        """Test that negative values return correct probabilities."""
        distribution = ZINB(**params)
        v = -1.0

        funcs_and_expected = [
            (distribution.cdf, 0.0),
            (distribution.pmf, 0.0),
            (distribution.log_pmf, -np.inf),
        ]

        for func, expected in funcs_and_expected:
            values = func(v)
            assert (np.asarray(values) == expected).all()

    @pytest.mark.parametrize("params", ZINB.get_test_params())
    def test_zinb_mean_positive(self, params):
        """Test that mean is non-negative."""
        distribution = ZINB(**params)
        mean = distribution.mean()
        assert (np.asarray(mean) >= 0).all()

    @pytest.mark.parametrize("params", ZINB.get_test_params())
    def test_zinb_var_positive(self, params):
        """Test that variance is non-negative."""
        distribution = ZINB(**params)
        var = distribution.var()
        assert (np.asarray(var) >= 0).all()

    @pytest.mark.parametrize("params", ZINB.get_test_params())
    def test_zinb_cdf_at_zero(self, params):
        """Test CDF at zero includes zero-inflation."""
        distribution = ZINB(**params)
        cdf_at_0 = distribution.cdf(0.0)
        # CDF at 0 should be >= pi (the zero-inflation probability)
        assert (np.asarray(cdf_at_0) >= params["pi"]).all()


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
class TestZeroInflated:
    """Tests for ZeroInflated compositor distribution."""

    @pytest.mark.parametrize("params", ZeroInflated.get_test_params())
    def test_zero_inflated_less_than_zero(self, params):
        """Test that negative values return correct probabilities."""
        distribution = ZeroInflated(**params)
        v = -1.0

        funcs_and_expected = [
            (distribution.cdf, 0.0),
            (distribution.pmf, 0.0),
            (distribution.log_pmf, -np.inf),
        ]

        for func, expected in funcs_and_expected:
            values = func(v)
            assert (np.asarray(values) == expected).all()

    @pytest.mark.parametrize("params", ZeroInflated.get_test_params())
    def test_zero_inflated_mean_positive(self, params):
        """Test that mean is non-negative."""
        distribution = ZeroInflated(**params)
        mean = distribution.mean()
        assert (np.asarray(mean) >= 0).all()

    @pytest.mark.parametrize("params", ZeroInflated.get_test_params())
    def test_zero_inflated_var_positive(self, params):
        """Test that variance is non-negative."""
        distribution = ZeroInflated(**params)
        var = distribution.var()
        assert (np.asarray(var) >= 0).all()

    @pytest.mark.parametrize("params", ZeroInflated.get_test_params())
    def test_zero_inflated_cdf_at_zero(self, params):
        """Test CDF at zero includes zero-inflation."""
        distribution = ZeroInflated(**params)
        cdf_at_0 = distribution.cdf(0.0)
        # CDF at 0 should be >= pi (the zero-inflation probability)
        assert (np.asarray(cdf_at_0) >= params["pi"]).all()


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
class TestZeroInflatedVsStandalone:
    """Test that ZeroInflated compositor gives same results as standalone."""

    def test_zero_inflated_equals_zipoisson(self):
        """Test ZeroInflated(Poisson) matches ZIPoisson."""
        from skpro.distributions import Poisson

        mu = 3.0
        pi = 0.3

        # Standalone ZIPoisson
        zi_pois = ZIPoisson(mu=mu, pi=pi)

        # Compositor version
        pois = Poisson(mu=mu)
        zi_comp = ZeroInflated(pi=pi, distribution=pois)

        # Test PMFs at various values
        for x in [0, 1, 2, 5, 10]:
            pmf_standalone = float(zi_pois.pmf(x))
            pmf_comp = float(zi_comp.pmf(x))
            np.testing.assert_allclose(pmf_standalone, pmf_comp, rtol=1e-10)

        # Test CDFs
        for x in [0, 1, 2, 5, 10]:
            cdf_standalone = float(zi_pois.cdf(x))
            cdf_comp = float(zi_comp.cdf(x))
            np.testing.assert_allclose(cdf_standalone, cdf_comp, rtol=1e-10)

        # Test mean
        mean_standalone = float(zi_pois.mean())
        mean_comp = float(zi_comp.mean())
        np.testing.assert_allclose(mean_standalone, mean_comp, rtol=1e-10)

        # Test variance
        var_standalone = float(zi_pois.var())
        var_comp = float(zi_comp.var())
        np.testing.assert_allclose(var_standalone, var_comp, rtol=1e-10)

    def test_zero_inflated_equals_zinb(self):
        """Test ZeroInflated(NegativeBinomial) matches ZINB."""
        from skpro.distributions import NegativeBinomial

        mu = 2.0
        alpha = 1.5
        pi = 0.25

        # Standalone ZINB
        zinb = ZINB(mu=mu, alpha=alpha, pi=pi)

        # Compositor version
        nb = NegativeBinomial(mu=mu, alpha=alpha)
        zi_comp = ZeroInflated(pi=pi, distribution=nb)

        # Test PMFs at various values
        for x in [0, 1, 2, 5, 10]:
            pmf_standalone = float(zinb.pmf(x))
            pmf_comp = float(zi_comp.pmf(x))
            np.testing.assert_allclose(pmf_standalone, pmf_comp, rtol=1e-10)

        # Test CDFs
        for x in [0, 1, 2, 5, 10]:
            cdf_standalone = float(zinb.cdf(x))
            cdf_comp = float(zi_comp.cdf(x))
            np.testing.assert_allclose(cdf_standalone, cdf_comp, rtol=1e-10)

        # Test mean
        mean_standalone = float(zinb.mean())
        mean_comp = float(zi_comp.mean())
        np.testing.assert_allclose(mean_standalone, mean_comp, rtol=1e-10)

        # Test variance
        var_standalone = float(zinb.var())
        var_comp = float(zi_comp.var())
        np.testing.assert_allclose(var_standalone, var_comp, rtol=1e-10)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
class TestParameterValidation:
    """Tests for parameter validation in zero-inflated distributions."""

    def test_zipoisson_invalid_pi_negative(self):
        """Test that negative pi raises ValueError."""
        with pytest.raises(ValueError, match="pi must be in"):
            ZIPoisson(mu=2.0, pi=-0.1)

    def test_zipoisson_invalid_pi_geq_one(self):
        """Test that pi >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="pi must be in"):
            ZIPoisson(mu=2.0, pi=1.0)

    def test_zipoisson_invalid_mu_negative(self):
        """Test that negative mu raises ValueError."""
        with pytest.raises(ValueError, match="mu must be positive"):
            ZIPoisson(mu=-1.0, pi=0.3)

    def test_zipoisson_invalid_mu_zero(self):
        """Test that zero mu raises ValueError."""
        with pytest.raises(ValueError, match="mu must be positive"):
            ZIPoisson(mu=0.0, pi=0.3)

    def test_zinb_invalid_pi_negative(self):
        """Test that negative pi raises ValueError."""
        with pytest.raises(ValueError, match="pi must be in"):
            ZINB(mu=2.0, alpha=1.0, pi=-0.1)

    def test_zinb_invalid_pi_geq_one(self):
        """Test that pi >= 1 raises ValueError."""
        with pytest.raises(ValueError, match="pi must be in"):
            ZINB(mu=2.0, alpha=1.0, pi=1.0)

    def test_zinb_invalid_mu_negative(self):
        """Test that negative mu raises ValueError."""
        with pytest.raises(ValueError, match="mu must be positive"):
            ZINB(mu=-1.0, alpha=1.0, pi=0.3)

    def test_zinb_invalid_alpha_negative(self):
        """Test that negative alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            ZINB(mu=2.0, alpha=-1.0, pi=0.3)

    def test_zinb_invalid_alpha_zero(self):
        """Test that zero alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            ZINB(mu=2.0, alpha=0.0, pi=0.3)

    def test_zero_inflated_invalid_pi_negative(self):
        """Test that negative pi raises ValueError."""
        from skpro.distributions import Poisson

        with pytest.raises(ValueError, match="pi must be in"):
            ZeroInflated(pi=-0.1, distribution=Poisson(mu=2.0))

    def test_zero_inflated_invalid_pi_geq_one(self):
        """Test that pi >= 1 raises ValueError."""
        from skpro.distributions import Poisson

        with pytest.raises(ValueError, match="pi must be in"):
            ZeroInflated(pi=1.0, distribution=Poisson(mu=2.0))

    def test_zero_inflated_invalid_pi_1d_array(self):
        """Test that 1D array pi raises ValueError."""
        from skpro.distributions import Poisson

        with pytest.raises(ValueError, match="pi must be a scalar or a 2D array"):
            ZeroInflated(pi=np.array([0.1, 0.2, 0.3]), distribution=Poisson(mu=2.0))

    def test_zero_inflated_invalid_pi_2d_shape_mismatch_rows(self):
        """Test that 2D pi with mismatched rows raises ValueError."""
        import pandas as pd

        from skpro.distributions import Poisson

        mu = np.array([[1.0, 2.0], [3.0, 4.0]])
        idx = pd.Index([0, 1])
        cols = pd.Index(["a", "b"])
        pois = Poisson(mu=mu, index=idx, columns=cols)

        # pi has 3 rows but distribution has 2
        with pytest.raises(ValueError, match="first dimension must match"):
            ZeroInflated(pi=np.array([[0.1], [0.2], [0.3]]), distribution=pois)

    def test_zero_inflated_invalid_pi_2d_shape_mismatch_cols(self):
        """Test that 2D pi with invalid column count raises ValueError."""
        import pandas as pd

        from skpro.distributions import Poisson

        mu = np.array([[1.0, 2.0], [3.0, 4.0]])
        idx = pd.Index([0, 1])
        cols = pd.Index(["a", "b"])
        pois = Poisson(mu=mu, index=idx, columns=cols)

        # pi has 3 columns but distribution has 2 (must be 1 or 2)
        with pytest.raises(ValueError, match="second dimension must be 1 or match"):
            ZeroInflated(
                pi=np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]), distribution=pois
            )


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
class TestBoundaryConditions:
    """Tests for boundary conditions of zero-inflated distributions."""

    def test_zipoisson_pi_zero(self):
        """Test that pi=0 reduces to standard Poisson."""
        from skpro.distributions import Poisson

        mu = 3.0
        zi_pois = ZIPoisson(mu=mu, pi=0.0)
        pois = Poisson(mu=mu)

        # Should match standard Poisson
        for x in [0, 1, 2, 5]:
            np.testing.assert_allclose(
                float(zi_pois.pmf(x)), float(pois.pmf(x)), rtol=1e-10
            )

    def test_zipoisson_pi_near_one(self):
        """Test that pi near 1 gives very high zero probability."""
        zi_pois = ZIPoisson(mu=2.0, pi=0.99)
        # P(0) should be very high
        assert float(zi_pois.pmf(0)) > 0.99

    def test_zinb_pi_zero(self):
        """Test that pi=0 reduces to standard NegativeBinomial."""
        from skpro.distributions import NegativeBinomial

        mu = 2.0
        alpha = 1.5
        zinb = ZINB(mu=mu, alpha=alpha, pi=0.0)
        nb = NegativeBinomial(mu=mu, alpha=alpha)

        # Should match standard NegativeBinomial
        for x in [0, 1, 2, 5]:
            np.testing.assert_allclose(float(zinb.pmf(x)), float(nb.pmf(x)), rtol=1e-10)

    def test_zinb_pi_near_one(self):
        """Test that pi near 1 gives very high zero probability."""
        zinb = ZINB(mu=2.0, alpha=1.5, pi=0.99)
        # P(0) should be very high
        assert float(zinb.pmf(0)) > 0.99

    def test_zero_inflated_pi_zero(self):
        """Test that pi=0 in ZeroInflated reduces to base distribution."""
        from skpro.distributions import Poisson

        mu = 3.0
        pois = Poisson(mu=mu)
        zi_pois = ZeroInflated(pi=0.0, distribution=pois)

        # Should match base distribution
        for x in [0, 1, 2, 5]:
            np.testing.assert_allclose(
                float(zi_pois.pmf(x)), float(pois.pmf(x)), rtol=1e-10
            )
