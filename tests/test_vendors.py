import pytest

from skpro.vendors.pymc import Pymc, PymcInterface


def test_construct_estimator():

    with pytest.raises(ValueError):
        Pymc()

    with pytest.raises(ValueError):
        # The model does not contain any free variables
        Pymc(model=PymcInterface(model_definition=lambda model, X, y: True))