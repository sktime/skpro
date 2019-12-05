import abc

from sklearn.base import BaseEstimator
from skpro.distributions.vectorization_mixin import VectorizationMixin

class DistributionBase(BaseEstimator, VectorizationMixin, metaclass=abc.ABCMeta):
    
    name_ = 'baseDistribution'

    @classmethod
    def name(self):
        return self.name_
    
    
    def _register(self):
 
        self.__checkInit()
        
        self.paramsDic_ = []
        
        if self.size_ > 1 :

            for i in range(self.size_):
                self.paramsDic_.append({})
                
            for key,value in self.get_params().items():
                
                if(not isinstance(value, list) or len(value) != self.size_):
                    raise ValueError('parameter [' + key + '] is not sized correctly, ' + str(self.n_) + ' expected')
    
                for i in range(self.size_) :
                    (self.paramsDic_[i])[key] = value[i]
        
        else : 
            self.paramsDic_.append(self.get_params())

            
    def __checkInit(self):  

        # size check
        if not hasattr(self, 'size_'):
             raise ValueError ('"size_" attribute must be instantiated before parameters registration')
        elif (self.size_ is None or self.size_ < 1):
             raise ValueError('"size_" attribute must be none zero before parameters registration')
        
        # dimension check
        if not hasattr(self, 'dimension_'):
             raise ValueError ('"dimension_" attribute must be instantiated before parameters registration')
        elif (self.dimension_ is None or self.dimension_ < 1):
             raise ValueError('"dimension_" attribute must be none zero before parameters registration')
            
       
    def get_params(self, index = None, deep=True):
        
        if index is None :
            return super(DistributionBase, self).get_params(deep)
        
        if index > self.size_ - 1 :
            raise ValueError('index for parameters call out of bound, (index:' + str(index) + ' > max:' + str(self.size_-1) + ')')
        
        return self.paramsDic_[index]
    
    

    # interface methods
    def pdf(self, y, mode = 'batch_wise'):
        return self._vectorize(self._pdfImp, y, mode)

    def pmf(self, y,  mode = 'batch_wise'):
        return self._vectorize(self._pmfImp, y, mode)

    def cdf(self, y,  mode = 'batch_wise'):
        return self._vectorize(self._cdfImp, y, mode)
    
    def squared_norm(self, mode = 'batch_wise'):
        return self._vectorize(self._sqrnImp, mode = mode)
    
    
    # impl methods
    def _pdfImp(self, y, args):
         raise ValueError('pdf function not implemented')

    def _pmfImp(self, y, args):
         raise ValueError('pmf function not implemented')

    def _cdfImp(self, y, args):
         raise ValueError('cdf function not implemented')
         
    def _sqrnImp(self, args):
         raise ValueError('squared norm function not implemented')

        

    
    

        
        
    


