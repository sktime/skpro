import numpy as np

class utils :
    
    def dim(a):
         """Utility method used to return the size of the input
         
         Parameters
         ----------
         X : array-like (list, nd.array) or scalar

         Returns
         -------
         dimension of X : int
         
         """

         if np.ndim(a) == 0:
             return  1
         elif(isinstance(a, list)):
             return len(a)
         elif type(a) is np.ndarray :
             return a.shape[0]
         else : raise ValueError('unrecognized type')
