import numpy as np
import abc

class LossFunction(metaclass=abc.ABCMeta):
     """Base interface class for classical (i.e. non probabilistic) Loss function :
     """

     type_ = 'classical'
        
     @staticmethod
     def type():
         return LossFunction.type_
        

class SquaredError(LossFunction) :
    """Squared Error loss function class
        i.e. for a target Yt and prediction Yp :
             Loss = (Yp - Yt)^2
     """
    
    def __call__(self, prediction_y, y) :
        """ () override that returns the loss evaluation asssociated with the following arguments
    
         Parameters
         ----------
         prediction_y : array of floats
             Array of estimated targets
              
         y : array of float
             Array of realized targets
             
         Returns
         ----------
         array of float
            Array of losses
         
         """
        
        return np.square(prediction_y - y)


def AbsoluteError(LossFunction):
    """Absolute Error loss function class
        i.e. for a target Yt and prediction Yp :
             Loss = abs(Yp - Yt)
     """
    
    def __call__(self, prediction_y, y) :
        """ () operator override that returns the loss evaluation asssociated with the following arguments
    
         Parameters
         ----------
         prediction_y : array of floats
             Array of estimated targets
              
         y : array of float
             Array of realized targets
             
         Returns
         ----------
         array of float
            Array of losses
         
         """
        return np.abs(prediction_y - y)

