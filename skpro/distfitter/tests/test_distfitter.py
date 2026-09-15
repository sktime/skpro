"""Tests for the distfitter module."""

from skpro.distfitter import MOMFitter
from skpro.distributions.normal import Normal


def test_get_params_deep_with_dist_cls():
    """get_params(deep=True) works when dist is a distribution class.

    Requires scikit-base>=1.0.1 (sktime/skbase#559).
    """
    fitter = MOMFitter(dist=Normal, mean_name="mu", std_name="sigma")
    params = fitter.get_params(deep=True)

    assert params["dist"] is Normal
    assert "dist__" not in params
