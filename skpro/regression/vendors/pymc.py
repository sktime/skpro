# -*- coding: utf-8 -*-
if False:
    from theano import shared
    import pymc3 as pm

from skpro.base.old_base import BayesianVendorInterface


class PymcInterface(BayesianVendorInterface):
    """PyMC3 interface

    Allows for the integration of PyMC3 models

    Parameters
    ----------
    model_definition: callable(model, X, y)
        Callable that defines a model using the
        given PyMC3 ``model`` variable and
        training features ``X`` as well as
        and the labels ``y``.
    samples_size: int (optional, default=500)
        Number of samples to be drawn from the
        posterior distribution
    """

    def __init__(self, model_definition, sample_size=500):
        self.model_definition = model_definition
        self.sample_size = sample_size
        self.model_ = pm.Model()
        self.X_ = None
        self.trace_ = None
        self.ppc_ = None

    def on_fit(self, X, y):
        self.X_ = shared(X)

        self.model_definition(model=self.model_, X=self.X_, y=y)

        with self.model_:
            self.trace_ = pm.sample()

    def on_predict(self, X):
        # Update the theano shared variable with test data
        self.X_.set_value(X)
        # Running PPC will use the updated values and do the prediction
        self.ppc_ = pm.sample_ppc(
            self.trace_, model=self.model_, samples=self.sample_size
        )

    def samples(self):
        return self.ppc_["y_pred"].T
