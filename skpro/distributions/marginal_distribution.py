import numpy as np
import math as m

from skpro.distributions.distribution_base import DistributionBase


class NormalDistribution(DistributionBase) :
    
     name_ = 'normal'
    
     def __init__(self, loc = 0.0, scale = 1.0):
        
        self.loc = loc 
        self.scale = scale 
        self.dimension_ = 1
        
        if isinstance(loc, list) :
            self.size_ = len(loc)
        else : self.size_ = 1
        
        
        self._register()
        
    
     def mean(self):
         return self.loc

 
     def _pdfImp(self, y, args):
        
         loc = args['loc']
         scale = args['scale']

         return np.exp(-(y - loc)**2/(2*scale))/np.sqrt(2*np.pi *scale)
    
     def _cdfImp(self, y, args):
         
         loc = args['loc']
         scale = args['scale']
        
         return 0.5  * (1.0 + m.erf((y - loc) / (np.sqrt(2.0 * scale))))
     
     def _sqrnImp(self, args):
         
         scale = args['scale']
                     
         return 1/(2**scale*np.sqrt(np.pi))
    
    
class LaplaceDistribution(DistributionBase) :
    
     name_ = 'laplace'
    
     def __init__(self, loc = 0.0, scale =1.0):
         
         self.loc = loc 
         self.scale = scale
         self.dimension_ = 1
         
         if isinstance(loc, list) :
            self.size_ = len(loc)
         else : self.size_ = 1
        
         self._register()
        
        
     def mean(self):
         return self.loc
 
    
     def _pdfImp(self, y, args):
         
         loc = args['loc']
         scale = args['scale']
         
         return  np.exp(-np.abs(y -loc)/scale)/(2*scale)
    
     def _cdfImp(self, y, args):
         loc = args['loc']
         scale = args['scale']
         
         if y < self.loc :
             return 0.5 * np.exp((y -loc)/scale)
         else:
             return 1 - 0.5 * np.exp(-(y -loc)/scale)
         
     def _sqrnImp(self, args):
         scale = args['scale']
         return 1/(4*scale)
