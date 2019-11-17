import numpy as np
import abc
from skpro.distributions.distribution import Distribution

   
class ProbabilisticLossFunction(metaclass=abc.ABCMeta) :
    
     def __init__(self):
        self.type = 'probabilistic'
        
     def check_density(self,f):
        if not isinstance(f, Distribution):
            raise ValueError("prediction entry is not a denstiy functor")
        pass
        
class LogLoss(ProbabilisticLossFunction) :

     def __call__(self, f, y) :
        self.check_density(f)
        return -np.log(f(y))


class LogLossClipped(ProbabilisticLossFunction) :
    
    def __init__(self, cap = np.exp(-23)):
        self.cap = cap

    def __call__(self, f, y) :
        self.check_density(f)
        return np.clip(a = -np.log(f(y)), a_max = -np.log(self.cap), a_min = None)
   
    
class IntegratedSquaredLoss(ProbabilisticLossFunction) :

    def __call__(self, f, y) :
        self.check_density(f)
        return - 2 * f(y) + f.squared_norm()
