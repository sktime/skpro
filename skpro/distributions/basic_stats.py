import abc

class BasicStatsMixin():
    """Mixin class for all the distribution class
    
    Notes
    -----
    Used by 'DistributionBase' class. 
    'DistributionBase' inherits a serie of 'BasicStats' abstract method 
    to be then instanciated in the concrete distribution sub-classes
     
    """
    @abc.abstractmethod
    def point(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def mean(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def variance(self):
        raise NotImplementedError()
  
    @abc.abstractmethod      
    def std(self):
        raise NotImplementedError()
        
    @abc.abstractmethod   
    def mode(self):
        raise NotImplementedError()