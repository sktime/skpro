import pytest

from skpro.vendors.pymc import Pymc, PymcInterface


def test_construct_estimator():

    with pytest.raises(ValueError):
        Pymc()

    model = Pymc(model=PymcInterface(model_definition=lambda model, X, y: True))

    assert isinstance(model, Pymc)