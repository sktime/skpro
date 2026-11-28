"""Tests for Normal survival and hazard in distribution tails."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from skpro.distributions.normal import Normal


def test_normal_surv_haz_finite_in_upper_tail():
    """Survival/hazard stay finite when cdf saturates at 1.0 (#1256)."""
    dist = Normal(mu=0, sigma=1)
    x = 12.0
    surv = dist.surv(x)
    haz = dist.haz(x)

    assert float(surv) > 0.0
    assert np.isfinite(float(haz))


def test_normal_surv_matches_scipy_sf_in_upper_tail():
    """Tail survival matches scipy.stats.norm.sf where cdf rounds to 1.0."""
    dist = Normal(mu=0, sigma=1)
    for x in (9.0, 12.0):
        assert float(dist.cdf(x)) == 1.0
        assert_allclose(float(dist.surv(x)), norm.sf(x), rtol=1e-12)


def test_normal_surv_matches_one_minus_cdf_in_interior():
    dist = Normal(mu=0, sigma=1)
    x = 0.5
    surv = dist.surv(x)
    cdf = dist.cdf(x)
    assert_allclose(float(surv), 1.0 - float(cdf), rtol=1e-10)


def test_normal_surv_array_finite_in_upper_tail():
    idx = pd.Index([0, 1])
    cols = pd.Index(["a"])
    dist = Normal(mu=[[0.0], [1.0]], sigma=1.0, index=idx, columns=cols)
    x = np.array([[12.0], [12.0]])
    surv = dist.surv(x)
    assert np.all(np.asarray(surv) > 0.0)
