from skpro.estimators.base import ProbabilisticEstimator
from skpro.baselines.adapters import DensityAdapter, KernelDensityAdapter


class DensityBaseline(ProbabilisticEstimator):

    def __init__(self, adapter=None):
        
        if adapter is None:
            adapter = KernelDensityAdapter()

        if not issubclass(adapter.__class__, DensityAdapter):
            raise ValueError('adapter has to be a subclass of skpro.density.DensityAdapter'
                             '%s given.' % adapter.__class__)

        self.adapter = adapter


    def fit(self, X, y, sample_weight=None):
        self.adapter(y)
        return self  

    def predict_proba(self, X):
        return self.adapter

    def score(self, X, y, sample=True, return_std=False):
        pass