"""Tests for Normal survival and hazard in distribution tails."""

import numpy as np
import pytest

from skpro.distributions.normal import Normal


def test_normal_surv_haz_finite_in_upper_tail():
    """Survival/hazard stay finite when cdf saturates at 1.0 (#1256)."""
    dist = Normal(mu=0, sigma=1)
    x = 12.0
    surv = dist.surv(x)
    haz = dist.haz(x)

    assert float(surv) > 0.0
    assert np.isfinite(float(haz))


def test_normal_surv_matches_one_minus_cdf_in_interior():
    dist = Normal(mu=0, sigma=1)
    x = 0.5
    surv = dist.surv(x)
    cdf = dist.cdf(x)
    np.testing.assert_allclose(float(surv), 1.0 - float(cdf), rtol=1e-10)