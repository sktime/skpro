
class BasicStatsMixin():
    """Mixin class for all the distribution class
    
    Notes
    -----
    Used by 'DistributionBase' class. 
    'DistributionBase' inherits a serie of 'BasicStats' abstract method 
    to be then instanciated in the concrete distribution sub-classes
     
    """
    
    def point(self):
        raise ValueError('mean function not implemented')

    def mean(self):
        raise ValueError('mean function not implemented')

    def variance(self):
        raise ValueError('variance function not implemented')
        
    def std(self):
        raise ValueError('variance function not implemented')
    
    def mode(self):
        raise ValueError('mode function not implemented')