import abc
import functools
import warnings
import numpy as np

from sklearn.base import BaseEstimator
from .metrics import log_loss


class ProbabilisticEstimator(BaseEstimator, metaclass=abc.ABCMeta):
    """
    Abstract base class for probabilistic prediction models
    """

    class ImplementsCachingAndCompatibility(abc.ABCMeta):
        """
        Enhances the distribution interface behind the
        scenes with automatic caching and ensures
        compatibility with other components
        """

        def __init__(cls, name, bases, clsdict):
            # Automatic caching of the interface methods
            for method in ['point', 'std']:
                cls._cached(cls, clsdict, method)

            # We generalize the std signature to allow for the
            # use with np.std() etc.
            def std(self, *args, **kwargs):
                return clsdict['std'](self)

            setattr(cls, 'std', std)

        def _cached(self, cls, clsdict, method):
            if method in clsdict:
                @functools.lru_cache()
                def cache_override(self, *args, **kwargs):
                    return clsdict[method](self, *args, **kwargs)

                # Override function
                setattr(cls, method, cache_override)

    class Distribution(metaclass=ImplementsCachingAndCompatibility):
        """
        Abstract base class for the distribution interface
        returned by probabilistic estimators
        """

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

        def pdf(self, x):
            warnings.warn(self.__class__.__name__ + ' does not implement a pdf function', UserWarning)

        def cdf(self, x):
            warnings.warn(self.__class__.__name__ + ' does not implement a cdf function', UserWarning)

        def ppf(self, x):
            warnings.warn(self.__class__.__name__ + ' does not implement a ppf function', UserWarning)

        def lp2(self):
            """
            Implements the L2 norm of the PDF

            ..math::
            L^2 = \int PDF(x)^2 dx

            :return:
            """
            warnings.warn(self.__class__.__name__ +
                          ' does not implement a lp2 function, defaulting to numerical approximation', UserWarning)

            def elementwise_integrand(func, index):
                """ Returns function for squared PDF at a certain index """
                def integrand(x):
                    return func(x)[index]**2

                return integrand

            from scipy.integrate import quad as integrate
            return np.array([
                # y, y_err of
                integrate(elementwise_integrand(self.pdf, index), -np.inf, np.inf)
                for index in range(len(self.X))
            ])[:, 0]

    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return '%s()' % self.__class__.__name__

    def __repr__(self):
        return '%s()' % self.__class__.__name__

    @classmethod
    def _distribution(cls):
        return cls.Distribution

    def predict(self, X):
        return self._distribution()(self, X)

    def fit(self, X, y):
        warnings.warn('The estimator doesn\'t implement a fit procedure', UserWarning)

        return self

    def score(self, X, y):
        return -1 * log_loss(self.predict(X), y, sample=True)
