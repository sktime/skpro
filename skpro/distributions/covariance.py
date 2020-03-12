
import numpy as np
import abc

class DeepObject(metaclass=abc.ABCMeta):
    """Deep object abstract class 
       Extends scikit-learn 'BaseEstimator' class 'deepness' functionality
       to non skpro non estimator object
       """
    @abc.abstractmethod
    def get_params(self):
        """abstract class 
        """
        raise NotImplementedError()
    

class CovarianceMatrix(DeepObject):
      """Covariance matrix class : 
          - Pass a raw covariance matrix (list of list) and process it as a ndarray 
          - do some basic assessement (symetry and strict positivity of the eingen values)
          - eventually store eingenvalues and eingenvectors should be stored in cach after first calculation.

        Parameters
        ----------
        cov: list of list of float
               contains the covariance entries

        frozen : boolean
             specify wether eingenvalues and eingenvectors should be stored in cach.
             Frozen set to TRUE corresponds to a cach mechanism activation.
             
        Note :
        ------

       """
    

      def __init__(self, cov, freeze = True):
          
          self.data_ = np.array(cov)
          self.freeze_ = freeze
          
          self.isDiagonal_ = False
          self.isDecomposed_ = False
          self.logdet_ = None
          self.__vals = None
          self.__vecs = None
    
          s = self.data_.shape

          if len(s) == 1:
              self.isDiagonal_ = True
          elif len(s) == 2 :
              if not np.allclose(self.data_, self.data_.T, 1e-05, 1e-08):
                  raise ValueError('cov is not symetric') 
          else :
              raise ValueError('entry wrongly formated')
              
          self.dim_ = s[0] 
              
              
      def __eq__(self, other): 
        if not isinstance(other, self.__class__):
            raise ValueError('cannot compare CovarianceMatrix to a non similar obect')
        
        return np.array_equal(self.data_, other.data_)
        
    
      def freeze(self):
          self.freeze_ = True
          pass
        
      def unfreeze(self):
           self.freeze_ = False
           self.isDecomposed_ = False
           pass
       
      def get_params(self):
          out = dict()
          out['data'] = self.cov()
          return out
      
      def cov(self):
          return self.data_.tolist()


      def __eig_decompo(self):
           """perform an EVD decomposition : 
               
           Returns :
               vals : list of float
                   eingenvalues, shape = (n_dimension, )
                   
               vecs : list of list of float
                   eingenvalues, shape = (n_dimension, n_dimension)

           Note :
           ------
           IF the decomposition is triggerd for the FIRST TIME and the freeze_ set to TRUE
              -> compute, stores the results and return
              
           IF the decomposition is triggerd after FIRST and the freeze_ set to TRUE
              -> only return the already computed results stored in the cach
           """
 
           if(not self.isDecomposed_ or not self.freeze_):
               
               if(self.isDiagonal_ ):
                    self.logdet_ = np.sum(np.log(self.data_))
               
               else:
                   self.__vals, self.__vecs = np.linalg.eig(self.data_)
                   
                   for i, v in enumerate(self.__vals):
                       if(v < 0) : raise ValueError('cov ' + str(i + 1) + ' is not positive definite') 

                   self.logdet_ = np.sum(np.log(self.__vals))
                

      def inverse(self):
          """return the Matrix inverse (from the EVD decompo) : 
               
           Returns :  
               ndarray, shape = (n_dimension, n_dimension)
          """
          if(self.isDiagonal_):
              return np.diag(1./self.data_)
          else:
              self.__eig_decompo()
              return np.dot(self.__vecs * 1./self.__vals, self.__vecs.transpose())
          
      
      def logdet(self):
          """return the log-determinent (from the EVD decompo) : 
               
           Returns :  
                float scalar
          """
          self.__eig_decompo()
          return self.logdet_

    
    