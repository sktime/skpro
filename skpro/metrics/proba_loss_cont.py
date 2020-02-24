
import numpy as np
from skpro.metrics.proba_loss_base import ProbabilisticLossFunction
from skpro.distributions.distribution_base import distType



class ProbabilisticContinuousLoss(ProbabilisticLossFunction) :
     """Base abstract class for the Probabilistic Loss function applied to continuous distribution :
     """
    
     def __init__(self):
         super().__init__()

     def check_distribution(self,f):
         """Method that check wether the argument f : 
             - is a DistributionBase class instance through the super() 'check_distribution' method.
             - is of continuous type
            and raise an error if FALSE.

         Parameters
         ----------
         f :  any 

        """
         super().check_distribution(f)
         
         if(f.dtype_ in [distType.DISCRETE, distType.UNDEFINED]):
            raise ValueError('pdf function not permitted for non continuous distribution')
         
         pass
    
    
class LogLoss(ProbabilisticContinuousLoss) :
     """Log loss probabilistic loss function class
        i.e. for a target Y and a density pdf(.) :
             Loss = log(pdf(Y))
     """

     def __call__(self, f, y) :
         """ () operator override that returns the loss evaluation asssociated with the following arguments
    
         Parameters
         ----------
         f : skpro distribution object (skpro.distribution.distribution_base)
              Define the distribution candidate to be evaluated
             
         y : array of float
             Array of realized targets
             
         Returns
         ----------
         array of float
              array of losses
         """
        
         self.check_distribution(f)

         return -np.log(f.pdf(y))


class LogLossClipped(ProbabilisticContinuousLoss) :
    """Capped Log loss probabilistic loss function class
       i.e. for a target Y and a density pdf(.) :
             Loss = min(log(pdf(Y)),CAP)
    
      Parameters
         ----------
         cap : scalar float
            defines the cap to be used 
     """
    
    def __init__(self, cap = np.exp(-23)):
         self.cap = cap
         

    def __call__(self, f, y) :
         """  () operator override that returns the loss evaluation asssociated with the following arguments
    
         Parameters
         ----------
         f : skpro distribution object (skpro.distribution.distribution_base)
              Define the distribution candidate to be evaluated
             
         y : array of float
             Array of realized targets
             
         Returns
         ----------
         array of float
            Array of losses
         """
         
         self.check_distribution(f)
        
         return np.clip(a = -np.log(f.pdf(y)), a_max = -np.log(self.cap), a_min = None)
   
    
class IntegratedSquaredLoss(ProbabilisticContinuousLoss) :
     """Log loss probabilistic loss function class
     i.e. for a target Y and a density pdf(.) :
             Loss = - 2 * pdf(y) + integral_{over the support}{pdf(s)^2 ds}
     """

     def __call__(self, f, y) :
         """ () operator override that returns the loss evaluation asssociated with the following arguments
    
         Parameters
         ----------
         f : skpro distribution object (skpro.distribution.distribution_base)
              Define the distribution candidate to be evaluated
             
         y : array of float
             Array of realized targets
             
         Returns
         ----------
         array of float
            Array of losses
         """
         
         self.check_distribution(f)
        
         return - 2 * f.pdf(y) + f.squared_norm()
