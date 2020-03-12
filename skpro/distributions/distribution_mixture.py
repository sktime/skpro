import numpy as np

from skpro.utils import utils
from skpro.distributions.distribution_base import DistributionBase, distType, Mode
from skpro.distributions.component.support import RealContinuousSupport

class Mixture(DistributionBase) :
     """Base class for the univariate normal distribution

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
    
    
     def __init__(self, distribution ,  weights):
         
        self.distribution = distribution
        self.weights = weights
        self.__check_args()
  
        super().__init__(
                name = 'mixture', 
                dtype = distType.CONTINUOUS,
                support = RealContinuousSupport(),
                vectorSize = utils.dim(distribution)
                )
        
        
     def __check_args(self):

         if isinstance(self.distribution, DistributionBase): 
             self.distribution.setMode(Mode.BATCH)
         elif isinstance(self.distribution, list) and isinstance(self.distribution[0], DistributionBase): 
             for d in self.distribution :
                 d.setMode(Mode.BATCH)
         else : raise ValueError('mixture distribution arg : argument must be of distribution type')
            
         if(not isinstance(self.weights, list)):
             raise ValueError('mixture weight arg : argument must be a list')
             
         if utils.dim(self.distribution) > 1 and utils.dim(self.weights[0])==1 :
             self.weights =  [self.weights for _ in range(len(self.weights))]
                    
     def pdf_imp(self, X):
         """Return the vectorized pdf of the mixture distribution
         
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
         w  = np.array(self.get_param('weights'))
         d  = self.get_param('distribution')

         dsize = utils.dim(d)
         out = [np.dot(np.array(d[i].pdf(X)), w[i]) for i in range(dsize)]
         out = np.array(out).T
         
         return out
     
        
     def cdf_imp(self, X):
         """Return the vectorized cdf of the mixture distribution
         
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
         w  = np.array(self.get_param('weights'))
         d  = self.get_param('distribution')

         dsize = utils.dim(d)
         out = [np.dot(np.array(d[i].cdf(X)), w[i]) for i in range(dsize)]
         out = np.array(out).T
         
         return out

    


    