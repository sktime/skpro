from theano import shared
import pymc3 as pm

from ..base import BayesianVendorEstimator, BayesianVendorInterface


class Pymc(BayesianVendorEstimator):

    def valid_model_definition(self):
        return callable(self.model.model_definition)


class PymcInterface(BayesianVendorInterface):

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
        self.ppc_ = pm.sample_ppc(self.trace_, model=self.model_, samples=self.sample_size)

    def samples(self):
        return self.ppc_['y_pred'].T


