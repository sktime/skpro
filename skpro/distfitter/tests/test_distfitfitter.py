"""Tests for the DistfitFitter distribution fitter."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd
import pytest

from skpro.distfitter import DistfitFitter
from skpro.distributions.base import BaseDistribution
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class(DistfitFitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_distfitfitter_fits_known_normal():
    """DistfitFitter restricted to 'norm' recovers close to true mean/scale."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.normal(loc=5.0, scale=2.0, size=1000))

    fitter = DistfitFitter(distr="norm")
    fitter.fit(X)

    assert fitter.dist_name_ == "norm"

    dist = fitter.proba()

    assert isinstance(dist, BaseDistribution)
    assert dist.ndim == 0

    mean = float(dist.mean())
    var = float(dist.var())

    assert np.isclose(mean, 5.0, atol=0.3)
    assert np.isclose(var, 4.0, atol=1.0)


@pytest.mark.skipif(
    not run_test_for_class(DistfitFitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_distfitfitter_stores_fit_summary():
    """DistfitFitter stores a summary table of all candidate distributions."""
    rng = np.random.RandomState(0)
    X = pd.DataFrame(rng.normal(size=200))

    fitter = DistfitFitter(distr=["norm", "expon"])
    fitter.fit(X)

    assert fitter.fit_summary_ is not None
    assert len(fitter.fit_summary_) == 2
    assert set(fitter.fit_summary_["name"]) == {"norm", "expon"}


@pytest.mark.skipif(
    not run_test_for_class(DistfitFitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_distfitfitter_proba_before_fit_raises():
    """Calling proba() before fit() must raise, per BaseDistFitter contract."""
    fitter = DistfitFitter(distr="norm")

    with pytest.raises(ValueError):
        fitter.proba()


@pytest.mark.skipif(
    not run_test_for_class(DistfitFitter),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_distfitfitter_get_params_roundtrip():
    """get_params/set_params/clone round-trip as expected for skpro estimators."""
    fitter = DistfitFitter(distr="norm", stats="wasserstein")
    params = fitter.get_params()

    assert params["distr"] == "norm"
    assert params["stats"] == "wasserstein"

    clone = fitter.clone()
    assert clone.get_params() == fitter.get_params()
