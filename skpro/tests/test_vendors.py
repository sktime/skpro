import pytest

from skpro.base.old_base import BayesianVendorEstimator
from skpro.regression.vendors.pymc import PymcInterface


def test_construct_estimator():

    with pytest.raises(ValueError):
        BayesianVendorEstimator()

    model = BayesianVendorEstimator(
        model=PymcInterface(model_definition=lambda model, X, y: True)
    )

    assert isinstance(model, BayesianVendorEstimator)