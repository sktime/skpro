
import numpy as np

class CovarianceMatrix:
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
    

      def __init__(self, cov, frozen = True):
          
          self.cov_ = np.array(cov)
          self.frozen_ = frozen
    
          self.isDecomposed_ = False
          self.vals_ = None
          self.vecs_ = None
          self.logdet_ = None
    
          s = self.cov_.shape
          if len(s) == 2:
              self.dim_ = 1
              self.dim_ = s[0] 
              
              if not np.allclose(self.cov_, self.cov_.T, 1e-05, 1e-08):
                  raise ValueError('cov is not symetric') 
          else :
              raise ValueError('entry wrongly formated')
              
              
      def __eq__(self, other): 
        if not isinstance(other, self.__class__):
            raise ValueError('cannot compare CovarianceMatrix to a non similar obect')
        
        return np.array_equal(self.cov_, other.cov_)
        
    
      def freeze(self):
          self.frozen_ = True
          pass
        
      def unfreeze(self):
           self.frozen_ = False
           self.isDecomposed_ = False
           pass


      def eig_decompo(self):
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
 
           if(not self.isDecomposed_):
               vals, vecs = np.linalg.eig(self.cov_)
               
               for i, v in enumerate(vals):
                   if(v < 0) : raise ValueError('cov ' + str(i + 1) + ' is not positive definite') 
               
               if self.frozen_ :
                   self.isDecomposed_ = True
                   self.vals_ = vals
                   self.vecs_ = vecs
                   self.logdet_ = np.sum(np.log(vals))
                   
               return vals, vecs
           
           else : return self.vals_, self.vecs_
        

      def inverse(self):
          """return the Matrix inverse (from the EVD decompo) : 
               
           Returns :  
               ndarray, shape = (n_dimension, n_dimension)
          """
          vals, vecs = self.eig_decompo()
          return np.dot(vecs * 1./vals, vecs.transpose())

    
    