import abc
import functools
import warnings
import numpy as np

from sklearn.base import BaseEstimator, clone
from .metrics import log_loss
from .density import DensityAdapter, KernelDensityAdapter
from .utils import ensure_existence


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


def _forward_meta(wrapper, f):
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
    wrapper.non_existing = getattr(f, 'not_existing', False)

    return wrapper


def _generalize(f):
    """ Generalizes the signature to allow for the use with np.std() etc.

    Parameters
    ----------
    f: The function to decorate

    Returns
    -------
    Decorated function
    """

    def wrapper(self, *args, **kwargs):
        return f(self)

    return _forward_meta(wrapper, f)


def _vectorize(f):
    """ Enables automatic vectorization of a function

    The wrapper vectorizes a interface function unless
    it is decorated with the vectorvalued decorator

    Parameters
    ----------
    f: The function to decorate

    Returns
    -------
    Decorated function
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

    return _forward_meta(wrapper, f)


def _elementwise(f):
    """ Enables elementwise operations

    The wrapper implements two different modes of argument evaluation
    for given p_1,..., p_k that represent the predicted distributions
    and and x_1,...,x_m that represent the values to evaluate them on.

    "elementwise" (default): Repeat the sequence of p_i until there are m,
                            i.e., p_1,...,p_k,p_1,p_2,...,p_k,p_1,...,p_m'
                            where m' is the remainder of dividing m by k.

    "batch": x_1, ..., x_m is evaluated on every distribution p_i
            resulting in a matrix m columns and k rows.

    Parameters
    ----------
    f: The function to decorate

    Returns
    -------
    Decorated function
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

    return _forward_meta(wrapper, f)


def _cached(f):
    """ Enables caching

    Wrapper uses lru_cache to cache function result

    Parameters
    ----------
    f: The function to decorate

    Returns
    -------
    Decorated function
    """

    @functools.lru_cache()
    def wrapper(self, *args, **kwargs):
        return f(self, *args, **kwargs)

    return _forward_meta(wrapper, f)


class ProbabilisticEstimator(BaseEstimator, metaclass=abc.ABCMeta):
    """ Abstract base class for probabilistic prediction models

    Notes
    -----
    All probabilistic estimators should specify all the parameters
    that can be set at the class level in their ``__init__``
    as explicit keyword arguments (no ``*args`` or ``**kwargs``).
    """

    class ImplementsEnhancedInterface(abc.ABCMeta):
        """ Meta-class for distribution interface

        Enhances the distribution interface behind the scenes
        with automatic caching and syntactic sugar for
        element-wise access of the distributions
        """

        def __init__(cls, name, bases, clsdict):
            for method in ['pdf', 'cdf']:
                if method in clsdict:
                    setattr(cls, method, _elementwise(ensure_existence(clsdict[method])))

            for method in ['point', 'std', 'lp2']:
                if method in clsdict:
                    setattr(cls, method, _cached(_vectorize(_generalize(ensure_existence(clsdict[method])))))

    class Distribution(metaclass=ImplementsEnhancedInterface):
        """
        Abstract base class for the distribution interface
        returned by probabilistic estimators

        Parameters
        ----------
        estimator: ``skpro.base.ProbabilisticEstimator``
            Parent probabilistic estimator object
        X: np.array
            Features
        selection: slice | int (optional)
            Subset point selection of the features
        mode: str
            Interface mode ('elementwise' or 'batch')
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
            """ Replicates the distribution object

            Parameters
            ----------
            selection: None | slice | int (optional)
                Subset point selection of the distribution copy
            mode: str (optional)
                Interface mode ('elementwise' or 'batch')

            Returns
            -------
            ``skpro.base.ProbabilisticEstimator.Distribution``
            """
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
            """ Variance prediction

            Returns
            -------
            The estimated standard deviation that corresponds to self.X
            """
            raise NotImplementedError()

        def pdf(self, x):
            """ Probability density function

            Parameters
            ----------
            x

            Returns
            -------
            mixed  Density function evaluated at x
            """
            warnings.warn(self.__class__.__name__ + ' does not implement a pdf function', UserWarning)

        def cdf(self, x):
            """ Cumulative density function

            Parameters
            ----------
            x

            Returns
            -------
            mixed Cumulative density function evaluated at x
            """
            warnings.warn(self.__class__.__name__ + ' does not implement a cdf function', UserWarning)

        def ppf(self, q, *args, **kwargs):
            """ Percent point function (inverse of cdf — percentiles).

            Parameters
            ----------
            q

            Returns
            -------
            float
            """
            warnings.warn(self.__class__.__name__ + ' does not implement a ppf function', UserWarning)

        def lp2(self):
            """
            Implements the Lp2 norm of the probability density function

            ..math::
            L^2 = \int PDF(x)^2 dx

            Returns
            -------
            float: Lp2-norm of the density function
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


###############################################################################


class VendorInterface(metaclass=abc.ABCMeta):
    """ Abstract base class for a vendor interface
    """

    def on_fit(self, X, y):
        """ Implements vendor fit procedure

        Parameters
        ----------
        X : np.array
            Training features
        y : np.array
            Training labels

        Returns
        -------
        None
        """
        pass

    def on_predict(self, X):
        """ Implements vendor predict procedure

        Parameters
        ----------
        X : np.array
            Test features

        Returns
        -------
        None
        """
        pass


class VendorEstimator(ProbabilisticEstimator):
    """ VendorEstimator

    ProbabilisticEstimator that interfaces a vendor using
    a VendorInterface and Adapter.

    Parameters
    ----------
    model: skpro.base.VendorInterface
        Vendor interface
    adapter: skpro.density.DensityAdapter
        Density adapter
    """

    class Distribution(ProbabilisticEstimator.Distribution,  metaclass=abc.ABCMeta):

        def cdf(self, x):
            ensure_existence(self.estimator.adapters_[self.index].cdf)

            return self.estimator.adapters_[self.index].cdf(x)

        def pdf(self, x):
            ensure_existence(self.estimator.adapters_[self.index].pdf)

            return self.estimator.adapters_[self.index].pdf(x)

    def __init__(self, model=None, adapter=None):
        self.model = self._check_model(model)
        self.adapter = self._check_adapter(adapter)
        self.adapters_ = []

    def _check_model(self, model=None):
        """ Checks the model

        Checks if vendor interface is valid

        Parameters
        ----------
        model: skpro.base.VendorInterface
            Vendor interface
        Returns
        -------
        skpro.base.VendorInterface
        """
        if not issubclass(model.__class__, VendorInterface):
            raise ValueError('model has to be a VendorInterface'
                             '%s given.' % model.__class__)

        return model

    def _check_adapter(self, adapter):
        """ Checks the adapter

        Can be overwritten to implement checking procedures for a
        density adapter that are applied during the object
        initialisation.

        Parameters
        ----------
        adapter: skpro.density.DensityAdapter
            Adapter

        Returns
        -------
        skpro.density.DensityAdapter
        """
        return adapter

    def fit(self, X, y):
        self.model.on_fit(X, y)

        return self

    def predict(self, X):
        self.model.on_predict(X)

        return super().predict(X)


class BayesianVendorInterface(VendorInterface):
    """ Abstract base class for a Bayesian vendor

    Notes
    -----
    Must implement the samples method that returns
    Bayesian posterior samples. The sample method
    should be cached using the ``functools.lru_cache``
    decorator to increase performance
    """

    @abc.abstractmethod
    @functools.lru_cache()
    def samples(self):
        raise NotImplementedError()


class BayesianVendorEstimator(VendorEstimator):
    """ Vendor estimator for Bayesian methods

    ProbabilisticEstimator that interfaces a Bayesian vendor using
    a BayesianVendorInterface and and sample-based Adapter.

    """

    class Distribution(VendorEstimator.Distribution):

        @vectorvalued
        def point(self):
            return self.estimator.model.samples().mean(axis=1)

        @vectorvalued
        def std(self):
            return self.estimator.model.samples().std(axis=1)

    def _check_model(self, model=None):
        if not issubclass(model.__class__, BayesianVendorInterface):
            raise ValueError('model has to be a subclass of skpro.base.BayesianVendorInterface'
                            '%s given.' % model.__class__)

        return model

    def _check_adapter(self, adapter=None):
        if adapter is None:
            # default adapter
            adapter = KernelDensityAdapter()

        if not issubclass(adapter.__class__, DensityAdapter):
            raise ValueError('adapter has to be a subclass of skpro.density.DensityAdapter'
                            '%s given.' % adapter.__class__)

        return adapter

    def predict(self, X):
        self.model.on_predict(X)

        # initialise adapter with samples
        samples = self.model.samples()
        for index in range(len(X)):
            adapter = clone(self.adapter)
            adapter(samples[index, :])
            self.adapters_.append(adapter)

        # return predicted distribution object
        return super().predict(X)


