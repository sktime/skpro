import abc
import numpy as np

from sklearn.neighbors import KernelDensity
from sklearn.base import clone


def ecdf(a):
    """ Returns the empirical distribution function of a sample

    Parameters
    ----------
    a: array
        Input array representing a sample

    Returns
    -------
    array xs   Empirical cdf of the input sample
    array ys
    """
    xs = np.sort(np.array(a))
    ys = np.arange(1, len(xs)+1)/float(len(xs))

    return xs, ys


def step_function(xs, ys):
    """
    Returns a step function from x-y pair sample

    Parameters
    ----------
    xs  x values
    ys  corresponding y values

    Returns
    -------
    function A step function
    """

    def func(x):
        index = np.searchsorted(xs, x)
        index = len(ys) - 1 if index >= len(ys) else index
        return ys[index]

    return func



class Adapter(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __call__(self, inlet):
        raise NotImplementedError()

    @abc.abstractmethod
    def pdf(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def cdf(self, x):
        raise NotImplementedError()


class KernelDensityAdapter(Adapter):

    def __init__(self, estimator=KernelDensity()):
        self.estimator = estimator
        self.estimators_ = []

    def __call__(self, inlet):
        """
        Adapter entry point

        Parameters
        ----------
        np.array(N, M) inlet: N bayesian samples of length M
        """

        # fit kernel density estimators for each row
        self.estimators_ = [
            clone(self.estimator).fit(inlet[index, :]) for index in range(len(inlet))
        ]

    def cdf(self, x):
        # TODO: How to obtain cdf that is consistent with the estimated PDF?
        pass

    def pdf(self, x):
        return [
            np.exp(self.estimators_[index].score_samples(x[:, np.newaxis]))[0]
            for index in range(len(self.estimators_))
        ]


class EmpiricalDensityAdapter(Adapter):

    def __init__(self):
        pass

    def __call__(self, inlet):
        pass

    def cdf(self, x):
        pass

    def pdf(self, x):
        pass