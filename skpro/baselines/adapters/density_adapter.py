import abc

import numpy as np
from scipy.integrate import simps
from sklearn.neighbors import KernelDensity

from skpro.distributions.distribution_base import DistributionBase, distType, Mode
from skpro.distributions.component.support import NulleSupport
from skpro.distributions.component.variate import VariateInfos
from skpro.utils import utils


class DensityAdapter(DistributionBase, metaclass=abc.ABCMeta):
    """
    Abtract base class for density adapter
    that transform an input into an
    density cdf/pdf interface
    """

    @abc.abstractmethod
    def __call__(self, inlet):
        """
        Adapter entry point

        Parameters
        ----------
        mixed inlet Input for the adapter transformation
        """
        raise NotImplementedError()
        

class KernelDensityAdapter(DensityAdapter):
    """
    DensityAdapter that uses kernel density estimation
    to transform samples
    """

    def __init__(self, estimator = None, bandwidth=0.2, cdf_grid_size = 1000, **kwargs):
        if estimator is None : 
            estimator = KernelDensity(bandwidth=bandwidth, **kwargs)
        self.estimator = estimator
        self.cdf_grid_size = cdf_grid_size
        
        super().__init__('KernelDensityAdapter', 
             dtype = distType.CONTINUOUS, vectorSize = 1,
             variateComponent = VariateInfos(),
             support = NulleSupport(), mode = Mode.BATCH)
        

    def __call__(self, sample):
        """
        Adapter entry point

        Parameters
        ----------
        sample : np.array inlet, shape = (n_samples, 1)
        """
        
        if len(sample.shape) > 1 :
            raise ValueError('Empircal adapter only implemented for 1.dim sample')
        
        # fit kernel density estimator
        self.minus_inf = min(sample) - 10 * np.std(sample)
        self.estimator.fit(sample[:, np.newaxis])


    def cdf_imp(self, X):
        """pdf
         
         Parameters
         ----------
         X : array-like, shape = (n_samples, 1)
            Test samples

         Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, 1) 
         """
        X = np.array(X)
        if(len(X.shape) > 1): raise ValueError('empirical adapter only implemented for 1.dim sample')

        if(utils.dim(X) == 1):
            grid_size = 1000
            step = (X - self.minus_inf) / grid_size
            grid = np.arange(self.minus_inf, X, step)
            pdf_estimation = np.exp(self.estimator.score_samples(grid.reshape(-1, 1)))
            return simps(y=pdf_estimation, dx=step)

        else: return np.array([self.cdf_imp(x) for x in X])


    def pdf_imp(self, X):
        """cdf
         
         Parameters
         ----------
         X : array-like, shape = (n_samples, 1)
            Test samples

         Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, 1)
        """
        
        X = np.array(X)
        if(len(X.shape) > 1): raise ValueError('empirical adapter only implemented for 1.dim sample')
        return np.exp(self.estimator.score_samples(X[:, np.newaxis]))


class EmpiricalDensityAdapter(DensityAdapter):
    """
    DensityAdapter that uses empirical cdf
    to transform samples
    """

    def __init__(self, bandwidth=0.2):
        self.xs_ = None
        self.n_ = None
        self.cfunction_ = None
        self.bandwidth_ = bandwidth
       
        super().__init__('EmpiricalDensityAdapter', 
             dtype =  distType.CONTINUOUS, vectorSize = 1,
             variateComponent = VariateInfos(),
             support = NulleSupport(), 
             mode = Mode.BATCH
        )
        

    def __call__(self, sample):
        """
        Adapter entry point

        Parameters
        ----------
        sample : np.array inlet, shape = (n_samples, 1)
        """
        if len(sample.shape) > 1 :
            raise ValueError('Empircal adapter only implemented for 1.dim sample')
        
        self.xs_ = np.sort(np.array(sample))
        self.n_ = len(self.xs_)
        
        def func(x):
            index = np.searchsorted(self.xs_, x, side = 'right')
            return index
        
        self.cfunction_ = func

    
    def cdf_imp(self, X):
         """pdf
         
         Parameters
         ----------
         X : array-like, shape = (n_samples, 1)
            Test samples

         Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, 1) 
         """
         X = np.array(X)
         if(len(X.shape) > 1): raise ValueError('empirical adapter only implemented for 1.dim sample')
         return self.cfunction_(X) /self.n_

    
    
    def pdf_imp(self, X):
        """pdf
         
         Parameters
         ----------
         X : array-like, shape = (n_samples, 1)
            Test samples

         Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, 1)
        """
        X = np.array(X)
        if(len(X.shape) > 1): raise ValueError('empirical adapter only implemented for 1.dim sample')
        
        w_ = self.bandwidth_ * 0.5
        num = self.cfunction_(X + w_) - self.cfunction_(X - w_)
        out = num/(self.bandwidth_ * self.n_)
        return out