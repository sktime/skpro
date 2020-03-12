import numpy as np

from skpro.utils import utils
from skpro.distributions.distribution_base import DistributionBase, distType
from skpro.distributions.component.support import  RealContinuousSupport


class Laplace(DistributionBase) :
     """Base class for the univariate Laplace distribution

       Parameters
        ----------
        loc: array-like or scalar (for 1-dim), shape = (m_distribution_size)
             vector of mean
        
        scale : array-like or scalar (for 1-dim), shape = (m_distribution_size)
             vector of standard-deviation
             
        Notes
        -----
        loc and scale list must be of same size

        """
    
    
     def __init__(self, loc = 0.0, scale = 1.0):
        
        self.loc = loc
        self.scale = scale 

        super().__init__(
                name = 'laplace', 
                dtype = distType.CONTINUOUS,
                vectorSize = utils.dim(loc),
                support = RealContinuousSupport()
                )


     def point(self):
        return self.loc
    
     def mean(self):
        return self.loc
    
     def std(self):
        return np.square(2) * self.scale

     def variance(self):
        return 2 * np.square(self.scale)
    
     def mode(self):
        return self.loc
    
    
     @classmethod    
     def varianceToScale(cls, variance):
         """Mapping of the distribution variance to the 'scale' parameter used in the __init__ method
         
         Notes: Used when a distribution prediction is made  (i.e. distribution object need to be instanciated)
         from a variance estimate
         
         Parameters
         ----------
         variance : array-like
            Test samples

         Returns
         -------
         Mapped 'scale' parameter : array of float

         """
         return np.sqrt(0.5*variance)
    

     def pdf_imp(self, X):
         """Return the vectorized pdf of the laplace distribution
         
         Parameters
         ----------
         X : array-like, shape = (n_samples)
            Test samples

         Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, m_distribution_size) in [BATCH-MODE]
                    (n_samples) in [ELEMENT-WISE-MODE]
         """
         
         loc = np.array(self.get_param('loc'))
         scale = np.array(self.get_param('scale'))

         m_ = utils.dim(loc)
         n_ = utils.dim(X)

         if(n_ > 1 and m_> 1):
             loc = np.array([loc] * n_)
             scale = np.array([scale] * n_)
             X = np.array([X] * m_).transpose()

         out = np.exp(-np.abs(X - loc)/scale)/(2*scale)

         return out
    
    
     def cdf_imp(self, X):
         """Return the vectorized cdf of the laplace distribution
         
         Parameters
         ----------
         X : array-like, shape = (n_samples)
            Test samples

         Returns
         -------
         cdf output : ndarray of float
            shape = (n_samples, m_distribution_size) in [BATCH-MODE]
                    (n_samples) in [ELEMENT-WISE-MODE]
         """

         loc = np.array(self.get_param('loc'))
         scale = np.array(self.get_param('scale'))

         m_ = utils.dim(loc)
         n_ = utils.dim(X)

         if(n_ > 1 and m_> 1):
             loc = np.array([loc] * n_)
             scale = np.array([scale] * n_)
             X = np.array([X] * m_).transpose()
             
         d_ = (X - loc)/scale
         bool_ = X < scale
         
         out = bool_ * 0.5 * np.exp(d_) + (1-bool_) * ( 1 - 0.5 * np.exp(-d_)) 
         
         return out


     def squared_norm_imp(self):
         """Return the vectorized L2 norm of the Laplace distribution

         Returns
         -------
         L2 norm output : ndarray of float
            shape = (m_distribution_size)

         """
         
         scale = np.array(self.get_param('scale'))
         out = 1/(4*scale)
                     
         return out
