"""LEGACY MODULE - TODO: remove or refactor."""

import abc
import functools
import warnings

import numpy as np
from sklearn.base import BaseEstimator, clone

from skpro.regression.density import DensityAdapter, KernelDensityAdapter
from skpro.utils.utils import ensure_existence


def vectorvalued(f):
    """Decorate a distribution function to disable automatic vectorization.

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
    """Forward meta information from decorated method to decoration.

    Parameters
    ----------
    wrapper
    f

    Returns
    -------
    Method with meta information
    """
    wrapper.already_vectorized = getattr(f, "already_vectorized", False)
    wrapper.non_existing = getattr(f, "not_existing", False)

    return wrapper


def _generalize(f):
    """Generalize the signature to allow for the use with np.std() etc.

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
    """Enable automatic vectorization of a function.

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

        if getattr(f, "already_vectorized", False):
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
    """Enable elementwise operations.

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
        elementwise = self.mode == "elementwise" and len(np.array(x).shape) != 0

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
    """Enable caching.

    Wrapper uses lru_cache to cache function result

    Parameters
    ----------
    f: The function to decorate

    Returns
    -------
    Decorated function
    """

    @functools.lru_cache
    def wrapper(self, *args, **kwargs):
        return f(self, *args, **kwargs)

    return _forward_meta(wrapper, f)


class ProbabilisticEstimator(BaseEstimator, metaclass=abc.ABCMeta):
    """Abstract base class for probabilistic prediction models.

    Notes
    -----
    All probabilistic estimators should specify all the parameters
    that can be set at the class level in their ``__init__``
    as explicit keyword arguments (no ``*args`` or ``**kwargs``).
    """

    class ImplementsEnhancedInterface(abc.ABCMeta):
        """Meta-class for distribution interface.

        Enhances the distribution interface behind the scenes
        with automatic caching and syntactic sugar for
        element-wise access of the distributions
        """

        def __init__(cls, name, bases, clsdict):
            for method in ["pdf", "cdf"]:
                if method in clsdict:
                    setattr(
                        cls, method, _elementwise(ensure_existence(clsdict[method]))
                    )

            for method in ["point", "std", "lp2"]:
                if method in clsdict:
                    setattr(
                        cls,
                        method,
                        _cached(
                            _vectorize(_generalize(ensure_existence(clsdict[method])))
                        ),
                    )

    class Distribution(metaclass=ImplementsEnhancedInterface):
        """Abstract base class for the distributions returned by estimators.

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

        def __init__(  # noqa
            self, estimator, X, selection=slice(None), mode="elementwise"  # noqa
        ):  # noqa
            self.estimator = estimator
            self._X = X
            self.index = slice(None)
            self.selection = selection
            if mode not in ["elementwise", "batch"]:
                mode = "elementwise"
            self.mode = mode

            if callable(getattr(self, "_init", None)):
                self._init()

        @property
        def X(self):
            """Test features.

            Reference of the test features that are ought to correspond
            with the predictive distribution represented by the interface.

            The interface methods (e.g. pdf) can use X to
            construct and exhibit the predictive distribution properties
            of the interface (e.g. construct the predicted pdf based on X)

            Note that X automatically reflects the feature point for which
            the interface is ought to represent the distributional
            prediction. For given M x n features, X will thus represent
            an 1 x n vector that provides the bases for the predicted
            distribution. However, if the :func:`.vectorvalued` decorator
            is applied X will represent the full M x n matrix for an
            efficient vectorized implementation.

            :getter: Returns the test features based on the current subset selection
            :setter: Sets the data reference
            :type: array
            """
            return self._X[self.selection, :][self.index]

        @X.setter
        def X(self, value):
            self._X = value

        def __len__(self):
            """Return the number of distributions represented by the interface."""
            shape = self.X.shape
            return shape[0] if len(shape) > 1 else 1

        def __setitem__(self, key, value):
            """Set a subset of the distribution object."""
            raise Exception("skpro distributions are readonly")

        def __delitem__(self, key):
            """Delete a subset of the distribution object."""
            raise Exception("skpro distributions are readonly")

        def replicate(self, selection=None, mode=None):
            """Replicatesthe distribution object.

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
            """Return a subset of the distribution object.

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
                if selection >= len(self):
                    raise IndexError("Selection is out of bounds")

                selection = slice(selection, selection + 1)

            # check for out of bounds subsets
            if len(range(*selection.indices(len(self)))) == 0:
                raise IndexError("Selection is out of bounds")

            # create subset replication
            replication = self.replicate(selection, mode)

            # rollback index
            self.index = index_

            return replication

        def __point__(self, name):
            """Point prediction."""
            if len(self) > 1:
                raise TypeError(
                    "Multiple distributions can not be converted to " + name
                )

            return self.point()

        def __float__(self):
            """Float prediction."""
            return float(self.__point__("float"))

        def __int__(self):
            """Int prediction."""
            return int(self.__point__("int"))

        @abc.abstractmethod
        def point(self):
            """Point prediction.

            Returns
            -------
            The point prediction that corresponds to self.X
            """
            raise NotImplementedError()

        def mean(self, *args, **kwargs):
            """Mean prediction.

            Returns
            -------
            The mean prediction that corresponds to self.X
            """
            return self.point()

        @abc.abstractmethod
        def std(self):
            """Variance prediction.

            Returns
            -------
            The estimated standard deviation that corresponds to self.X
            """
            raise NotImplementedError()

        def pdf(self, x):
            """Probability density function.

            Parameters
            ----------
            x

            Returns
            -------
            mixed  Density function evaluated at x
            """
            warnings.warn(  # noqa
                self.__class__.__name__ + " does not implement a pdf function",
                UserWarning,
            )

        def cdf(self, x):
            """Cumulative density function.

            Parameters
            ----------
            x

            Returns
            -------
            mixed  Cumulative density function evaluated at x
            """
            warnings.warn(  # noqa
                self.__class__.__name__ + " does not implement a cdf function",
                UserWarning,
            )

        def ppf(self, q, *args, **kwargs):
            """Percent point function (inverse of cdf — percentiles).

            Parameters
            ----------
            q

            Returns
            -------
            float
            """
            warnings.warn(  # noqa
                self.__class__.__name__ + " does not implement a ppf function",
                UserWarning,
            )

        def lp2(self):
            r"""Compute Lp2 norm of the probability density function.

            ..math::
            L^2 = \int PDF(x)^2 dx

            Returns
            -------
            float: Lp2-norm of the density function
            """
            warnings.warn(  # noqa
                f"{self.__class__.__name__} "
                "does not implement a lp2 function, "
                "defaulting to numerical approximation",
                UserWarning,
            )

            from scipy.integrate import quad as integrate

            # y, y_err of
            return integrate(lambda x: self[self.index].pdf(x) ** 2, -np.inf, np.inf)[0]

    def name(self):
        """Return the name of the estimator."""
        return self.__class__.__name__

    def __str__(self):
        """Return the name of the estimator."""
        return "%s()" % self.__class__.__name__

    def __repr__(self):
        """Return the repr of the estimator."""
        return "%s()" % self.__class__.__name__

    @classmethod
    def _distribution(cls):
        return cls.Distribution

    def predict(self, X):
        """Predict using the model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = (n_samples, n_features)
            Samples.

        Returns
        -------
        :class:`.Distribution` interface representing n_samples predictions
            Returns predicted distributions
        """
        return self._distribution()(self, X)

    def fit(self, X, y):
        """Fit the model.

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
        warnings.warn(  # noqa
            "The estimator doesn't implement a fit procedure", UserWarning  # noqa
        )  # noqa

        return self  # noqa

    def score(self, X, y, sample=True, return_std=False):
        """Return the log-loss score.

        Parameters
        ----------
            X:  np.array
                Features
            y:  np.array
                Labels
            sample: boolean, default=True
                If true, loss will be averaged across the sample
            return_std: boolean, default=False
                If true, the standard deviation of the
                loss sample will be returned

        Returns
        -------
        mixed
            Log-loss score
        """
        return make_scorer(log_loss, greater_is_better=False)(  # noqa
            self, X, y, sample=sample, return_std=return_std
        )


###############################################################################


class VendorInterface(metaclass=abc.ABCMeta):  # noqa
    """Abstract base class for a vendor interface."""

    def on_fit(self, X, y):  # noqa
        """Vendor fit procedure.

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

    def on_predict(self, X):  # noqa
        """Vendor predict procedure.

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
    """VendorEstimator.

    ProbabilisticEstimator that interfaces a vendor using
    a VendorInterface and Adapter.

    Parameters
    ----------
    model: skpro.base.VendorInterface
        Vendor interface
    adapter: skpro.density.DensityAdapter
        Density adapter
    """

    class Distribution(ProbabilisticEstimator.Distribution, metaclass=abc.ABCMeta):
        """Distribution class returned by VendorEstimator.predict(X)."""

        pass

    def __init__(self, model=None, adapter=None):
        """Construct self.

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
        """Check the model.

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
            raise ValueError(
                "model has to be a VendorInterface" "%s given." % model.__class__
            )

        return model

    def _check_adapter(self, adapter):
        """Check the adapter.

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
        """Fit the vendor model.

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
        """Predict using the vendor model.

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
    """Abstract base class for a Bayesian vendor.

    Notes
    -----
    Must implement the samples method that returns
    Bayesian posterior samples. The sample method
    should be cached using the ``functools.lru_cache``
    decorator to increase performance
    """

    @abc.abstractmethod
    @functools.lru_cache  # noqa
    def samples(self):
        """Return the predictive posterior samples.

        Returns
        -------
        np.array
            Predictive posterior sample
        """
        raise NotImplementedError()


class BayesianVendorEstimator(VendorEstimator):
    """Vendor estimator for Bayesian methods.

    ProbabilisticEstimator that interfaces a Bayesian vendor using
    a BayesianVendorInterface and and sample-based Adapter.

    """

    class Distribution(VendorEstimator.Distribution):
        """Distribution class returned by BayesianVendorEstimator.predict(X)."""

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
            """Point prediction."""
            return self.samples.mean(axis=1)

        @vectorvalued
        def std(self):
            """Std prediction."""
            return self.samples.std(axis=1)

        def cdf(self, x):
            """Cumulative density function.

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
            """Probability density function.

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
            raise ValueError(
                "model has to be a subclass of skpro.base.BayesianVendorInterface"
                "%s given." % model.__class__
            )

        return model

    def _check_adapter(self, adapter=None):
        if adapter is None:
            # default adapter
            adapter = KernelDensityAdapter()

        if not issubclass(adapter.__class__, DensityAdapter):
            raise ValueError(
                "adapter has to be a subclass of skpro.density.DensityAdapter"
                "%s given." % adapter.__class__
            )

        return adapter
