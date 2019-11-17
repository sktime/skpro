import numpy as np
import abc


class LossFunction(metaclass=abc.ABCMeta):
    def __init__(self):
        self.type = 'classical'
    

class SquaredError(LossFunction) :
    
    def __call__(self, prediction_y, y) :
        return np.square(prediction_y - y)


def AbsoluteError(LossFunction):
    
    def __call__(self, prediction_y, y) :
        return np.abs(prediction_y - y)

