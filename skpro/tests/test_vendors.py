import pytest

if False:
    from skpro.base import BayesianVendorEstimator
    from skpro.vendors.pymc import PymcInterface


def test_construct_estimator():

    with pytest.raises(ValueError):
        BayesianVendorEstimator()

    model = BayesianVendorEstimator(
        model=PymcInterface(model_definition=lambda model, X, y: True)
    )

    assert isinstance(model, BayesianVendorEstimator)