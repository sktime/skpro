import numpy as np

from skpro.distributions.distribution_base import DistributionBase, distType
from skpro.distributions.component.support import  RealContinuousSupport
from skpro.distributions.component.variate import VariateInfos
from skpro.distributions.covariance import CovarianceMatrix


class MultiVariateNormal(DistributionBase) :
     """Base class for the multivariate normal distribution

         Parameters
         ----------
         loc: array-like, shape = (m_distribution_size, d_dimension)
             arrays of means
        
         scale : array-like, shape = (m_distribution_size, d_dimension, d_dimension)
             arrays of covariance
             
         Notes
         -----
         Distributions are instanciated as vector of distribution if
         a 2D array for loc (3D for scal respectively) are passed.
         The covariance array is processed into a 'CovarianceMatrix' object within the __init__ method
            
     """

     def __init__(self, loc, cov):
         
         if not isinstance(loc, list):
              raise ValueError('parameter [loc] is not sized correctly, a list is expected')

         loc_size_ = np.array(loc).shape
         
         if(len(loc_size_) == 1):
             dimension = loc_size_[0]
             size = 1
             self.cov = CovarianceMatrix(cov, freeze = True)
         else :
             dimension = loc_size_[1]
             size = loc_size_[0]
             self.cov = [CovarianceMatrix(c, freeze = True) for c in cov]

         self.loc = loc

         super().__init__('multivariate.normal', 
             distType.CONTINUOUS, 
             vectorSize = size, 
             variateComponent = VariateInfos(dimension),
             support = RealContinuousSupport()
          )
    
    
     def point(self):
        return self.loc
         
     def mean(self):
        return self.loc

     def variance(self):
        return self.cov
    
     def mode(self):
        return self.loc
    

     def pdf_imp(self, X):
         """Return the vectorized pdf of the multivariate normal distribution
         
         Parameters
         ----------
         X : array-like, shape = (n_samples, d_features)
            Test samples

         Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, m_distribution_size) in [BATCH-MODE]
                    (n_samples) in [ELEMENT-WISE-MODE]
            
         """

         loc = np.array(self.get_param('loc'))
         cov = np.array(self.get_param('cov'))
         
         #check dimension
         if not isinstance(X, list):
            raise ValueError('input not sized correctly, a list is expected')
         
         loc_shape = np.array(loc).shape
         m_ = loc_shape[0] if len(loc_shape) > 1 else 1
 
         X = np.array(X)
         x_shape = X.shape

         if(len(x_shape) > 1):
             n_ = x_shape[0]
             d_ = x_shape[1]
         else:
             n_ = 1
             d_ = x_shape[0]

         if(d_ != self.variateSize()):
             raise ValueError('X dimension do not match the distribution dimension')

         #process the cache of the cov
         if(m_ == 1) : cov = [cov.item()]
         
         logdet = np.array([c.logdet() for c in cov])
         log2pi = np.log(2 * np.pi)
         
         # compute the vectorized mean difference (i.e. x - loc in (n_ x m_) )
         if(n_ > 1):
             X_ = [[e] * m_ for e in X]
             d_ = np.array([loc] * n_) - np.array(X_)
         else :
             d_ = [loc] - np.array([X] * m_)
  
         # compute the vectorized quadratic form (n_ x m_)
         maha = np.array([(d_[:, i].dot(cov[i].inverse())*d_[:, i]).sum(axis=1) for i in range(m_)])
         logdet = np.array([logdet.transpose()] * n_)
         
         out = np.exp(-0.5 * (maha.transpose() +  self.variateSize() * log2pi + logdet))
         
         if(n_ == 1 or m_ == 1):
             out = out.flatten()

         return out