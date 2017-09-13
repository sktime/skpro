import abc
from sklearn.base import BaseEstimator
import functools
import warnings
from .metrics import log_loss


def describe(estimator):
    """
    Returns a description for a given estimator

    Args:
        estimator (Estimator): Estimator

    Returns:
        str: Estimator description
    """
    if isinstance(estimator, str):
        return estimator

    # Custom description?
    if getattr(estimator, 'description', False):
        return estimator.description()
    # Pipeline?
    elif estimator.__class__.__name__ == 'Pipeline':
        r = '{'
        sep = ''
        for name, model in estimator.steps:
            r += sep + name
            sep = ', '
        r += '}'
        return r
    # Fall-through default
    else:
        return estimator.__class__.__name__


class ProbabilisticEstimator(BaseEstimator, metaclass=abc.ABCMeta):

    class InterfaceCache(abc.ABCMeta):
        """
        Automatic caching for the interface methods
        """
        def __init__(cls, name, bases, clsdict):
            for method in ['point', 'std']:
                cls._cached(cls, clsdict, method)

            # adjust std signature to allow for the
            # use with np.std etc.
            def std(self, *args, **kwargs):
                return clsdict['std'](self)

            setattr(cls, 'std', std)

        def _cached(self, cls, clsdict, method):
            if method in clsdict:
                @functools.lru_cache()
                def cache_override(self, *args, **kwargs):
                    return clsdict[method](self, *args, **kwargs)

                # override function
                setattr(cls, method, cache_override)

    class Distribution(metaclass=InterfaceCache):

        def __init__(self, estimator, X):
            self.estimator = estimator
            self.X = X

        def __len__(self):
            return len(self.X)

        def __getitem__(self, key):
            return self.point()[key]

        def __setitem__(self, key, value):
            raise Exception('skpro interfaces are readonly')

        def __delitem__(self, key):
            raise Exception('skpro interfaces are readonly')

        @abc.abstractmethod
        def point(self):
            raise NotImplementedError()

        @abc.abstractmethod
        def std(self):
            raise NotImplementedError()

        @abc.abstractmethod
        def pdf(self, x):
            raise NotImplementedError()

        def cdf(self, x):
            warnings.warn(self.__name__ + ' does not implement a cdf function', UserWarning)

        def ppf(self, x):
            warnings.warn(self.__name__ + ' does not implement a ppf function', UserWarning)

        def lp2(self):
            warnings.warn(self.__name__ + ' does not implement a lp2 function', UserWarning)

    def name(self):
        return self.__class__.__name__

    def __repr__(self):
        return '%s()' % self.__class__.__name__

    @classmethod
    def _distribution(cls):
        return cls.Distribution

    def predict(self, X):
        return self._distribution()(self, X)

    def fit(self, X, y):
        self.estimators.fit(X, y)

        return self

    def score(self, X, y):
        return -1 * log_loss(self.predict(X), y, sample=True)
