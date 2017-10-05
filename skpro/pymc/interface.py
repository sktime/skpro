import abc
import functools

from theano import shared
import pymc3 as pm


class ImplementsSampleCaching(abc.ABCMeta):
    """
    Enables automatic caching of the interface sample
    """

    def __init__(cls, name, bases, clsdict):
        if 'samples' in clsdict:
            @functools.lru_cache()
            def cache_override(self, *args, **kwargs):
                return clsdict['samples'](self, *args, **kwargs)
            setattr(cls, 'samples', cache_override)


class InterfacePyMC(metaclass=ImplementsSampleCaching):

    def __init__(self):
        self.model = pm.Model()
        self.X = None
        self.trace = None
        self.ppc = None

    def on_fit(self, X, y):
        pass

    def on_predict(self, X):
        pass

    @abc.abstractmethod
    def samples(self):
        raise NotImplementedError()


class PlugAndPlayPyMC(InterfacePyMC):

    def __init__(self, model_definition=None):
        self.model_definition = model_definition
        super().__init__()

    def on_fit(self, X, y):
        self.X = shared(X)

        self.model_definition(model=self.model, X=self.X, y=y)

        with self.model:
            self.trace = pm.sample()

    def on_predict(self, X):
        # Update the theano shared variable with test data
        self.X.set_value(X)
        # Running PPC will use the updated values and do the prediction
        self.ppc = pm.sample_ppc(self.trace, model=self.model, samples=500)

    def samples(self):
        return self.ppc['y_pred'].T