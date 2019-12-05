import abc
#import functools
import numpy as np

from sklearn.base import BaseEstimator


#def vectorvalued(f):
#    """ Decorates a distribution function to disable automatic vectorization.
#
#    Parameters
#    ----------
#    f: The function to decorate
#
#    Returns
#    -------
#    Decorated function
#    """
#    f.already_vectorized = True
#    return f
#
#
#def _forward_meta(wrapper, f):
#    """ Forward meta information from decorated method to decoration
#
#    Parameters
#    ----------
#    wrapper
#    f
#
#    Returns
#    -------
#    Method with meta information
#    """
#    wrapper.already_vectorized = getattr(f, 'already_vectorized', False)
#    wrapper.non_existing = getattr(f, 'not_existing', False)
#
#    return wrapper
#
#
#def _generalize(f):
#    """ Generalizes the signature to allow for the use with np.std() etc.
#
#    Parameters
#    ----------
#    f: The function to decorate
#
#    Returns
#    -------
#    Decorated function
#    """
#
#    def wrapper(self, *args, **kwargs):
#        return f(self)
#
#    return _forward_meta(wrapper, f)
#
#
#def _vectorize(f):
#    """ Enables automatic vectorization of a function
#
#    The wrapper vectorizes a interface function unless
#    it is decorated with the vectorvalued decorator
#
#    Parameters
#    ----------
#    f: The function to decorate
#
#    Returns
#    -------
#    Decorated function
#    """
#    def wrapper(self, *args, **kwargs):
#        # cache index
#        index_ = self.index
#        self.index = slice(None)
#
#        if getattr(f, 'already_vectorized', False):
#            result = f(self, *args, **kwargs)
#        else:
#            result = []
#            for index in range(len(self.X)):
#                self.index = index
#                result.append(f(self, *args, **kwargs))
#
#        # rollback index
#        self.index = index_
#
#        if len(result) > 1:
#            return np.array(result)
#        else:
#            return result[0]
#
#    return _forward_meta(wrapper, f)
#
#
#def _elementwise(f):
#    """ Enables elementwise operations
#
#    The wrapper implements two different modes of argument evaluation
#    for given p_1,..., p_k that represent the predicted distributions
#    and and x_1,...,x_m that represent the values to evaluate them on.
#
#    "elementwise" (default): Repeat the sequence of p_i until there are m,
#                            i.e., p_1,...,p_k,p_1,p_2,...,p_k,p_1,...,p_m'
#                            where m' is the remainder of dividing m by k.
#
#    "batch": x_1, ..., x_m is evaluated on every distribution p_i
#            resulting in a matrix m columns and k rows.
#
#    Parameters
#    ----------
#    f: The function to decorate
#
#    Returns
#    -------
#    Decorated function
#    """
#
#    def wrapper(self, x, *args, **kwargs):
#        if len(np.array(x).shape) > 1:
#            x = x.flatten()
#
#        # cache index
#        index_ = self.index
#        self.index = slice(None)
#
#        # disable elementwise mode if x is scalar
#        elementwise = (self.mode == 'elementwise' and len(np.array(x).shape) != 0)
#
#        if elementwise:
#            evaluations = len(x)
#        else:
#            evaluations = len(self.X)
#
#        # compose result
#        result = []
#        number_of_points = len(self.X)
#        for index in range(evaluations):
#            # set evaluation index and point
#            if elementwise:
#                self.index = index % number_of_points
#                at = x[index]
#            else:
#                self.index = index
#                at = x
#
#            # evaluate the function at this point
#            result.append(f(self, at, *args, **kwargs))
#
#        # rollback index
#        self.index = index_
#
#        if len(result) > 1:
#            return np.array(result)
#        else:
#            return result[0]
#
#    return _forward_meta(wrapper, f)
#
#
#def _cached(f):
#    """ Enables caching
#
#    Wrapper uses lru_cache to cache function result
#
#    Parameters
#    ----------
#    f: The function to decorate
#
#    Returns
#    -------
#    Decorated function
#    """
#
#    @functools.lru_cache()
#    def wrapper(self, *args, **kwargs):
#        return f(self, *args, **kwargs)
#
#    return _forward_meta(wrapper, f)
#

class ProbabilisticEstimator(BaseEstimator, metaclass=abc.ABCMeta):
    """ Abstract base class for probabilistic prediction models

    Notes
    -----
    All probabilistic estimators should specify all the parameters
    that can be set at the class level in their ``__init__``
    as explicit keyword arguments (no ``*args`` or ``**kwargs``).
    """
    
    def __init__(self, estimator, X, selection=slice(None), mode='elementwise'):
            self.estimator = estimator
            self._X = X
            self.index = slice(None)
            self.selection = selection
            if mode not in ['elementwise', 'batch']:
                mode = 'elementwise'
            self.mode = mode

            if callable(getattr(self, '_init', None)):
                self._init()

    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return '%s()' % self.__class__.__name__

    def __repr__(self):
        return '%s()' % self.__class__.__name__
    

    def predict(self, X):
        distribution = self.predict_proba(X = X)
        return distribution.mean()
    
    def predict_proba(self, X):
        
        raise ValueError('The estimator doesn\'t implement a predict_proba procedure')
        
        pass

    


    def fit(self, X, y):
        raise ValueError('The estimator doesn\'t implement a fit procedure')

        return self

    def score(self, X, y, sample=True, return_std=False):

        raise ValueError('The estimator doesn\'t implement a score procedure')
        
        pass



###############################################################################


'''class VendorInterface(metaclass=abc.ABCMeta):
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

        pass

    def __init__(self, model=None, adapter=None):
        """

        Parameters
        ----------
        model : :class:`.VendorInterface`
            The vendor model
        adapter :class:`.DensityAdapter`
            Used density adapter
        """
        self.model = self._check_model(model)
        self.adapter = self._check_adapter(adapter)

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
        """
        Fits the vendor model

        Parameters
        ----------
        X : numpy array or sparse matrix of shape [n_samples,n_features]
            Training data
        y : numpy array of shape [n_samples, n_targets]
            Target values. Will be cast to X's dtype if necessary

        Returns
        -------
        self : returns an instance of self.
        """
        self.model.on_fit(X, y)

        return self

    def predict(self, X):
        """Predicts using the vendor model

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        :class:`.Distribution` interface representing n_samples predictions
            Returns predicted distributions
        """
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
        """
        Returns the predictive posterior samples

        Returns
        -------
        np.array
            Predictive posterior sample
        """
        raise NotImplementedError()


class BayesianVendorEstimator(VendorEstimator):
    """ Vendor estimator for Bayesian methods

    ProbabilisticEstimator that interfaces a Bayesian vendor using
    a BayesianVendorInterface and and sample-based Adapter.

    """

    class Distribution(VendorEstimator.Distribution):

        def _init(self):
            # initialise adapter with samples
            self.adapters_ = []
            self.samples = self.estimator.model.samples()
            for index in range(len(self.X)):
                adapter = clone(self.estimator.adapter)
                adapter(self.samples[index, :])
                self.adapters_.append(adapter)

        @vectorvalued
        def point(self):
            return self.samples.mean(axis=1)

        @vectorvalued
        def std(self):
            return self.samples.std(axis=1)

        def cdf(self, x):
            """ Cumulative density function

            Parameters
            ----------
            x

            Returns
            -------
            mixed  Cumulative density function evaluated at x
            """
            ensure_existence(self.adapters_[self.index].cdf)

            return self.adapters_[self.index].cdf(x)

        def pdf(self, x):
            """ Probability density function

            Parameters
            ----------
            x

            Returns
            -------
            mixed  Density function evaluated at x
            """
            ensure_existence(self.adapters_[self.index].pdf)

            return self.adapters_[self.index].pdf(x)

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
        '''