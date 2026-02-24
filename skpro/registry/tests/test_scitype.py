"""Tests for scitype typing function."""

import pytest

from skpro.registry._scitype import scitype


@pytest.mark.parametrize("coerce_to_list", [True, False])
def test_scitype(coerce_to_list):
    """Test that the scitype function recovers the correct scitype(s)."""
    from skpro.distributions.laplace import Laplace
    from skpro.distributions.normal import Normal
    from skpro.regression.bayesian import BayesianConjugateLinearRegressor
    from skpro.regression.mapie import MapieRegressor
    from skpro.regression.residual import ResidualDouble

    # test that scitype works for classes with soft dependencies
    result_mapie = scitype(MapieRegressor, coerce_to_list=coerce_to_list)
    if coerce_to_list:
        assert isinstance(result_mapie, list)
        assert "regressor_proba" == result_mapie[0]
    else:
        assert "regressor_proba" == result_mapie

    # test that scitype works for Bayesian regressor
    result_bayesian = scitype(BayesianConjugateLinearRegressor, coerce_to_list=coerce_to_list)
    if coerce_to_list:
        assert isinstance(result_bayesian, list)
        assert "regressor_proba" == result_bayesian[0]
    else:
        assert "regressor_proba" == result_bayesian

    # test that scitype works for instances
    inst = ResidualDouble.create_test_instance()
    result_naive = scitype(inst, coerce_to_list=coerce_to_list)
    if coerce_to_list:
        assert isinstance(result_naive, list)
        assert "regressor_proba" == result_naive[0]
    else:
        assert "regressor_proba" == result_naive

    # test distribution object
    result_laplace = scitype(Laplace, coerce_to_list=coerce_to_list)
    if coerce_to_list:
        assert isinstance(result_laplace, list)
        assert "distribution" == result_laplace[0]
    else:
        assert "distribution" == result_laplace

    # test Normal distribution object (part of Bayesian API)
    result_normal = scitype(Normal, coerce_to_list=coerce_to_list)
    if coerce_to_list:
        assert isinstance(result_normal, list)
        assert "distribution" == result_normal[0]
    else:
        assert "distribution" == result_normal