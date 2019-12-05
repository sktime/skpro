import numpy as np
import math as m

from skpro.distributions.distribution_base import DistributionBase


class MultiVariateNormal(DistributionBase) :
    
     name_ = 'normal'

     def __init__(self, loc, cov):
         
         if not isinstance(loc, list):
              raise ValueError('parameter [loc] is not sized correctly, a list is expected')

         tmp = np.shape(loc)
         
         if(len(tmp) == 1):
             self.dimension_ = tmp[0]
             self.size_ = 1
         else :
             self.dimension_ = tmp[1]
             self.size_ = tmp[0]

         self.loc = loc
         self.cov = cov  
         
         self._register()
         
         
     def mean(self):
         return self.loc


     def _pdfImp(self, y, args):
         
         if not isinstance(y, list):
            raise ValueError('input not sized correctly, a list is expected')
            
         if len(y) != self.dimension_ :
            raise ValueError('input list not sized correctly, (size:' + str(len(y)) 
             + ', expect:' + str(self.dimension_-1) + ')')
            
         loc = args['loc']
         cov = args['cov']
         
         det = np.linalg.det(cov)
         if det == 0:
            raise ValueError("det of covariance is singular")
            
         nfactor = det * (2 * m.pi) ** self.dimension_
         nfactor = 1 / m.sqrt(nfactor)

         d_ = np.array(y) - np.array(loc)
         invcov_ = np.linalg.inv(np.array(cov))
         prod = np.dot(np.dot(d_, invcov_), d_.T)
         
         return nfactor * np.exp(- 0.5 * prod)

