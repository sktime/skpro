from skpro.distributions.distribution_base import DistributionBase, distType
from skpro.distributions.component.support import NulleSupport



class CustomDistribution(DistributionBase) :
     """Customized distribution class :
        The pdf and cdf can be defined dynamically by the user (using lambda functions)

        Parameters
        ----------
        name : string
             distribution tag string
             
        pdf_func : function
             lambda function that serves as pdf
             
        cdf_func : function
             lambda function that serves as cdf
             
        support : support object
             distribution support object (skpro.distributions.component.Support)
         
        func_args : kargs for additional parameters to be used for the cdf and pdf function
             

        """

     def __init__(self, name = None, pdf_func = None, cdf_func = None, support = NulleSupport(), **func_args):
        
         self.pdf_func_ = pdf_func
         self.cdf_func_ = cdf_func
         self.func_args_ = func_args
        
         super().__init__(name, distType.CONTINUOUS, vectorSize = 1, 
             support = support)


     def pdf_imp(self, X):
         """Return the vectorized pdf of the custom distribution
         
         Parameters
         ----------
         X : array-like, shape = (n_samples)
            Test samples

         Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, m_distribution_size) in [BATCH-MODE]
                    (n_samples) in [ELEMENT-WISE-MODE]

         """
         
         
         if(self.pdf_func_ is None):
             raise ValueError('pdf functor not implemented')

         args = self.func_args_

         return self.pdf_func_(X, **args)
    
    
     def cdf_imp(self, X):
         """Return the vectorized cdf of the normal distribution
         
         Parameters
         ----------
         X : array-like, shape = (n_samples)
            Test samples

         Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, m_distribution_size) in [BATCH-MODE]
                    (n_samples) in [ELEMENT-WISE-MODE]

         """
         if(self.cdf_func_ is None):
             
             raise ValueError('pdf functor not implemented')
             
         args = self.func_args_
         
         return self.cdf_func_(X, **args)
    
     

    
