# LEGACY MODULE - TODO: remove or refactor

import pytest

from skpro.base.old_base import BayesianVendorEstimator
from skpro.regression.vendors.pymc import PymcInterface


@pytest.mark.skip(reason="avoiding pymc3 dependency for now")
def test_construct_estimator():
    with pytest.raises(ValueError):
        BayesianVendorEstimator()

    model = BayesianVendorEstimator(
        model=PymcInterface(model_definition=lambda model, X, y: True)
    )

    assert isinstance(model, BayesianVendorEstimator)
