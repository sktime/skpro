
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

    def __init__(self, form =  variateEnum.univariate, size = None):
        self.form_ = form
        self.size_ = size
        
        if(self.form_ is self.variateEnum.univariate) :
            self.size_ = 1
        