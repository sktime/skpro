
import abc
from skpro.distributions.distribution_base import DistributionBase

   
class ProbabilisticLossFunction(metaclass=abc.ABCMeta) :
     """Base Abstract class for the Probabilistic Loss function concrete class

     """
     
     type_ = 'probabilistic'
        
     @staticmethod
     def type():
         return ProbabilisticLossFunction.type_
    


     def check_distribution(self,f):
        """Method that check wether the argument f is a DistributionBase class instance
         (raise an error if not). Mean to be used in all the Probabilistic Loss sub concrete class

         Parameters
         ----------
         f :  any 

        """

        if not isinstance(f, DistributionBase):
            raise ValueError("prediction entry is not a denstiy functor")
        pass 