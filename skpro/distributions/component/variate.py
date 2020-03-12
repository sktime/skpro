
from enum import Enum

class VariateInfos :
    
    '''VariateInfos
     store in a structure the dimension informations of a distribution class in a nested way 
     (contains a dimension size and variateEnum [UNIVARIATE, MULTIVARIATE])
     
     To be instantiated for each distribution class.
     (by default through the default VariateInfos() instanciation passed into the distributionBase class)
    '''
    
    class variateEnum(Enum):
        univariate = 1
        multivariate = 2

    def __init__(self, size = 1):
        self.size_ = size
        
        self.form_ = self.variateEnum.univariate
        if(self.size_ > 1) : self.form_ = self.variateEnum.multivariate

        