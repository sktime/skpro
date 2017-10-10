import abc
import functools
import warnings
import numpy as np

from sklearn.base import BaseEstimator
from .metrics import log_loss


def vectorvalued(f):
    """ Decorates a distribution function to disable automatic vectorization.

    Parameters
    ----------
    f: The function to decorate

    Returns
    -------
    Decorated function
    """
    f.already_vectorized = True
    return f


def _with_meta(wrapper, f):
    """ Forward meta information from decorated method to decoration

    Parameters
    ----------
    wrapper
    f

    Returns
    -------
    Method with meta information
    """
    wrapper.already_vectorized = getattr(f, 'already_vectorized', False)
    return wrapper


def _generalize(f):
    """ Generalizes the signature to allow for the use with np.std() etc.
    """

    def wrapper(self, *args, **kwargs):
        return f(self)

    return _with_meta(wrapper, f)


def _vectorize(f):
    """ Enables automatic vectorization of a function

    The wrapper vectorizes a interface function unless
    it is decorated with the vectorvalued decorator
    """
    def wrapper(self, *args, **kwargs):
        # cache index
        index_ = self.index
        self.index = slice(None)

        if getattr(f, 'already_vectorized', False):
            result = f(self, *args, **kwargs)
        else:
            result = []
            for index in range(len(self.X)):
                self.index = index
                result.append(f(self, *args, **kwargs))

        # rollback index
        self.index = index_

        if len(result) > 1:
            return np.array(result)
        else:
            return result[0]

    return _with_meta(wrapper, f)


def _elementwise(f):
    """ Enables elementwise operations

    The wrapper implements two different modes of argument evaluation
    for given p_1,...,p_k that represent the predicted distributions
    and and x_1,...,x_m that represent the values to evaluate them on.

    "elementwise" (default): Repeat the sequence of p_i until there are m,
                            i.e., p_1,...,p_k,p_1,p_2,...,p_k,p_1,...,p_m'
                            where m' is the remainder of dividing m by k.

    "batch": x_1, ..., x_m is evaluated on every distribution p_i
            resulting in a matrix m columns and k rows.
    """

    def wrapper(self, x, *args, **kwargs):
        if len(np.array(x).shape) > 1:
            x = x.flatten()

        # cache index
        index_ = self.index
        self.index = slice(None)

        # disable elementwise mode if x is scalar
        elementwise = (self.mode == 'elementwise' and len(np.array(x).shape) != 0)

        if elementwise:
            evaluations = len(x)
        else:
            evaluations = len(self.X)

        # compose result
        result = []
        number_of_points = len(self.X)
        for index in range(evaluations):
            # set evaluation index and point
            if elementwise:
                self.index = index % number_of_points
                at = x[index]
            else:
                self.index = index
                at = x

            # evaluate the function at this point
            result.append(f(self, at, *args, **kwargs))

        # rollback index
        self.index = index_

        if len(result) > 1:
            return np.array(result)
        else:
            return result[0]

    return _with_meta(wrapper, f)


def _cached(f):
    """ Enables caching

    Wrapper uses lru_cache to cache function result
    """

    @functools.lru_cache()
    def wrapper(self, *args, **kwargs):
        return f(self, *args, **kwargs)

    return _with_meta(wrapper, f)


class ProbabilisticEstimator(BaseEstimator, metaclass=abc.ABCMeta):
    """
    Abstract base class for probabilistic prediction models
    """

    class ImplementsEnhancedInterface(abc.ABCMeta):
        """
        Enhances the distribution interface behind the scenes
        with automatic caching and syntactic sugar for
        element-wise access of the distributions
        """

        def __init__(cls, name, bases, clsdict):
            for method in ['pdf', 'cdf']:
                if method in clsdict:
                    setattr(cls, method, _elementwise(clsdict[method]))

            for method in ['point', 'std', 'lp2']:
                if method in clsdict:
                    setattr(cls, method, _vectorize(clsdict[method]))

    class Distribution(metaclass=ImplementsEnhancedInterface):
        """
        Abstract base class for the distribution interface
        returned by probabilistic estimators
        """

        def __init__(self, estimator, X, selection=slice(None), mode='elementwise'):
            self.estimator = estimator
            self._X = X
            self.index = slice(None)
            self.selection = selection
            if mode not in ['elementwise', 'batch']:
                mode = 'elementwise'
            self.mode = mode

        @property
        def X(self):
            return self._X[self.selection, :][self.index]

        @X.setter
        def X(self, value):
            self._X = value

        def __len__(self):
            shape = self.X.shape
            return shape[0] if len(shape) > 1 else 1

        def __setitem__(self, key, value):
            raise Exception('skpro distributions are readonly')

        def __delitem__(self, key):
            raise Exception('skpro distributions are readonly')

        def replicate(self, selection=None, mode=None):
            if selection is None:
                selection = self.selection

            if mode is None:
                mode = self.mode

            return self.__class__(self.estimator, self._X, selection, mode)

        def __getitem__(self, key):
            """Returns a subset of the distribution object

            Parameters
            ----------
            - slice indexing, mode (optional)
            - mode only (in which full subset is returned)

            Returns
            -------
            ``skpro.base.ProbabilisticEstimator.Distribution``
            """

            # cache index
            index_ = self.index
            self.index = slice(None)

            # parse key
            if isinstance(key, tuple) and len(key) == 2:
                selection = key[0]
                mode = key[1]
            elif isinstance(key, str):
                selection = slice(None)
                mode = key
            else:
                selection = key
                mode = None

            # convert index to slice for consistent usage
            if isinstance(selection, int):
                if selection > 40:
                    a = 1

                if selection >= len(self):
                    raise IndexError('Selection is out of bounds')

                selection = slice(selection, selection+1)

            # check for out of bounds subsets
            if len(range(*selection.indices(len(self)))) == 0:
                raise IndexError('Selection is out of bounds')

            # create subset replication
            replication = self.replicate(selection, mode)

            # rollback index
            self.index = index_

            return replication

        def __point__(self, name):
            if len(self) > 1:
                raise TypeError('Multiple distributions can not be converted to ' + name)

            return self.point()

        def __float__(self):
            return float(self.__point__('float'))

        def __int__(self):
            return int(self.__point__('int'))

        @abc.abstractmethod
        def point(self):
            """ Point prediction

            Returns
            -------
            The point prediction that corresponds to self.X
            """
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

            from scipy.integrate import quad as integrate
                   # y, y_err of
            return integrate(lambda x: self[self.index].pdf(x)**2, -np.inf, np.inf)[0]

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
        return -1 * log_loss(y, self.predict(X), sample=True)
