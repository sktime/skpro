
import numpy as np


class BaseSet :
    """Set class to be used as main composite class for the support class
       The BaseSet defined a sup and inf boundaries and implements 
       a 'inSet' method that returns a single bolean wether the all samples 
       in argument 'X' are within the defined boundaries

       Parameters
       ----------
       sup : float scalar
               sup boundary of the set
               
       inf : float scalar
               inf boundary of the set

    """
    
    def __init__(self, sup = None, inf = None):
        self.sup_ = sup
        self.inf_ = inf
        
   
    def _process(self, X):
        
        if np.isscalar(X):
            X = np.array([X])
        elif isinstance(X, list):
            X = np.array(X)
            
        return X
        
        
    def inSet(self, X):
        """Returns wether all samples in argument 'X' are within the defined boundaries
         
         Parameters
         ----------
         X : array-like, shape = (n_samples)
            Test samples

         Returns
         -------
         output : single boolean, TRUE if all n_samples belongs to the defined set, FALSE else

         """
        
        X = self._process(X)

        if(self.sup_ is not None and any(X > self.sup_)):
            return False
        
        if(self.inf_ is not None and any(X < self.inf_)):
            return False
        
        return True
    
    
    
    
    
class DiscreteSet(BaseSet) :
    """
       sub 'Set' class that defines the acceptable range as an predifined list
       a 'inSet' method that returns a bolean wether the samples 'X' is within the defined boundaries

       Parameters
       ----------
       discreteSet: list of float
                defined the discrete range of acceptable element
  
       """
    
    def __init__(self, discreteSet = []):
        self.discreteSet_ = discreteSet
        sup = max(self.discreteSet_)
        inf = min(self.discreteSet_)
        super().__init__(sup, inf)
        
        

    def inSet(self, X):
        """Returns wether all samples in argument 'X' are within the defined boundaries
         
         Parameters
         ----------
         X : array-like, shape = (n_samples)
            Test samples

         Returns
         -------
         output : single boolean, TRUE if all n_samples belongs to the defined set, FALSE else

         """

        if(all(self._process(X) in self.discreteSet_)):
                return True
            
        return False
    
    
    
class CompositeSet(BaseSet) :
    """
       sub 'Set' class that defines the acceptable range as a list of already instantiated sets object

       Parameters
       ----------
       inclusionSets: list of Set object 
                  list of sets defining the acceptable range 
                  (i.e. any evaluation point will be evaluated postivily if tested positively to any of these Set objects)
       
       exclusionSets: list of Set object
                 list of sets defining an exclusion range 
                  (i.e. any evaluation point will be evaluated negatively if tested positively to any of these Set objects)

       """
    
    def __init__(self, inclusionSets, exclusionSets = []):
        self.exclusionSets_ = exclusionSets
        self.inclusionSets_ = inclusionSets
        
        inf = float("inf")
        sup = float("-inf")
        
        if isinstance(inclusionSets, list) :

            for s in self.inclusionSets_ :
                sup = max(sup, s.sup_)
                inf = min(inf, s.inf_)

        super().__init__(sup, inf)
        

    def inSet(self, X):
        """Returns wether all samples in argument 'X' are within the defined boundaries
         
         Parameters
         ----------
         X : array-like, shape = (n_samples)
            Test samples

         Returns
         -------
         output : single boolean, TRUE if all n_samples belongs to the defined set, FALSE else

         """
        
        X = self._process(X)
        
        for s in self.exclusionSets_:
            if(s.inSet(X)):
                return False
        
        for s in self.inclusionSets_:
            if(not s.inSet(X)):
                return False
            
        return True

    