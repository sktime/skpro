
from skpro.distributions.component.set import BaseSet


class Support :
    """Support class to be passed within each distribution class. - 
       To be used to ensure that test samples points belongs to the range of the distribution support. 

       Parameters
       ----------
       supportSet: Set object (skpro.distributions.component.set) 
       
       Notes
       ------------
       The support class works as a wrap-up class of the Suport class. 
       More specific support type can be instantiate by inheriting from support and enforcing specific set type underneath  

    """

    def __init__(self, supportSet = BaseSet()):
        self.set_ =  supportSet

    def set(self) : 
        return self.set_

        
    def inSupport(self, X) :
       """Support main method that return wether the 

       Parameters
       ----------
       X : array-like, shape = (n_samples)
            Test samples
            
       Returns
        -------
        Boolean : TRUE if X is in the set range FALSE else

       """
       
       return self.set_.inSet(X)
    


class RealContinuousSupport(Support) :
    """Support class for the continuous support over the real
    """
    
    def __init__(self) :
        super().__init__(
                supportSet = BaseSet(float("inf"), float("-inf"))
                )
        
    def inSupport(self, X) : 
        return True
    
    
        
class NulleSupport(Support) :
    """Nulle support used by default in the DistributionBase constructor
       'inSupport' returns True for all 'X' samples
    """
    
    def __init__(self) :
        super().__init__(
                supportSet = BaseSet(float("inf"), float("-inf"))
                )
    
    def inSupport(self, X) : 
        return True


    
