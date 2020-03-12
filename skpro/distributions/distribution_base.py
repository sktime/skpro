import abc

import numpy as np
from enum import Enum
from sklearn.base import BaseEstimator

from skpro.distributions.component.support import NulleSupport
from skpro.distributions.component.variate import VariateInfos

from skpro.distributions.basic_stats import BasicStatsMixin
from skpro.distributions.dpqr import DPQRMixin
from skpro.utils import utils


class distType(Enum):
        UNDEFINED = 0
        DISCRETE = 1
        CONTINUOUS = 2
        MIXED = 3
        
class Mode(Enum):
        ELEMENT_WISE = 0
        BATCH = 1
         
      
  
class DistributionBase(BaseEstimator, BasicStatsMixin, DPQRMixin, metaclass=abc.ABCMeta):
    """Base Abstract class for the distribution sub-classes

        Parameters
        ----------
        name: string 
             distribution string tag
        
        dtype : distType Enum
             distribution type (undefined by default)
             
        vectorSize : int
             distribution vector size (1 by default)
             
        variateComponent : VariateInfos struct (skpro.distributions.component.VariateInfos)
             store in a structure the dimension informations (int size and variateEnum [UNIVARIATE, MULTIVARIATE])
             
        support : support object
             distribution support object (skpro.distributions.component.Support)
             To be used to assess the validity of the evaluation point passed to the pdf/pmf/cdf methods (through its methods inSupport(.)). 
             By default the NulleSupport() instantites a 'inSupport(.)' method that returns TRUE for all argument.
             
        mode : Mode Enum
             specify the pdf/pmf/cdf methods evaluation mode by default
             
        Note :
        ------
            
            For a vectorized distribution object, evaluation functions (cdf, pdf, pmf, ...) can be called in two different mode. 
            Assuming a m.size distribution object and a n.size samples of evaluation point :
                
                1. [BATCH] evaluation mode [active by-default] : evaluates on a each-for-each basis, 
                   i.e. returns a nxm matrix output if (n > 1) or a mx1 vector if (n = 1)
                   
                2. [ELEMENT_WISE] evaluation mode evaluates on a one for one basis. 
                   It repeats the sequence of distribution p_i until there are m, i.e., p_1,...,p_n,p_1,p_2,...,p_n,p_1,...,p_m' 
                   where m is the remainder of dividing m by n. Thus will output a m sized array

    """
      

    def __init__(self, name, 
                 dtype = distType.UNDEFINED, 
                 vectorSize = 1, 
                 variateComponent = VariateInfos(), 
                 support = NulleSupport(),
                 mode = Mode.BATCH
         ):

        self.name_ = name
        self.dtype_ = dtype
        self.cached_index_ = slice(None)
        
        self.vectorSize_ = vectorSize
        self.variateComponent_ = variateComponent
        self.support_ = support
        
        if type(mode) is not Mode : mode = Mode.ELEMENT_WISE
        self.mode_ = mode


  
    #member accessor
    def name(self):
        return self.name_
    
    def parameters(self):
        return self.paramsFrame_.data()
    
    def support(self):
        return self.support_
    
    def dtype(self):
        return self.dtype_
   
    def variateSize(self):
        return self.variateComponent_.size_
    
    def variateForm(self):
        return self.variateComponent_.form_.name
    
    def vectorSize(self):
        return self.vectorSize_
    
    def reset(self):
        self.cached_index_ = slice(None)
        
    def getMode(self):
        return self.mode_


    def setMode(self, mode):
        """reset the pdf/pmf/cdf methods evaluation mode

        Parameters
        ----------        
        mode : Mode Enum
             specify the pdf/pmf/cdf methods evaluation mode

        """
        
        if type(mode) is not Mode :
            raise ValueError('unrecognize distribution mode')

        self.mode_ = mode
        

    
    def get_params(self, index = None, deep=True):
        """overriden implementation of the scikit-learn 'BaseEstimator'. 
        If the distribution is vectorized, passing an index enables to return a dictionary of parameters 
        for the indexed distribution only. If no index is set then the super() implementation of get_params() is called 
        (i.e. returning all set parameters).

        Parameters
        ----------
        index : integer
             index of the distribution whose parameters are returned
             
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.
      
         Returns
         -------
            ``parameters dictionary of the type {string, values}``
        """
        
        if index is None :
            out = dict()

            for key in self._get_param_names():
                value = getattr(self, key, None)
                out[key] = value
            
            if(deep): out = DistributionBase.__addDeepAttributes(out)
            return out

        else : 
            return self.__getitemParams(index, deep)


    def get_param(self, key):
        """ private method that return a list containing the keyed parameter. 
        If the distribution is vectorized it will only operates for the distribution subset indexed by 'cached_index'.
            
        Parameters
        ----------
        key : string
             key parameter to be returned
             
         Returns
         -------
            ``parameters list for the distribution indexed by "cached_index"
            
         Note
         -------
         It is meant to be used in the pdf/pmf/cdf implied methods of the concrete distribution class to call the parameters for evaluation. 
         It will thus only evaluates the distribution subset indexed by the 'cached-index'. The 'cached-index' can thus be pre-modified to adapt to the different mode of evaluation
         By default 'cached_index' is set (and reset) to 'slice(None)' which corresponds to a BATCH evaluation where all distributions are evaluated
        
        """
        
        if not isinstance(key, str):
             raise ValueError('key index must be a parameter string')

        #return self.paramsFrame_.getParameter(key)[self.cached_index_]
        out = self.get_params(deep = False)[key]
        
        if(utils.dim(out) == 1) : 
            return out
        else :
            return out[self.cached_index_]
    
    
    def __getitemParams(self, index = None, deep = 'False'):

        out = dict()
        
        for key, value in self.get_params(deep = False).items() :
            if(utils.dim(value) == 1) : out[key] = value
            else : out[key] = value[index]
            
        if deep : out = DistributionBase.__addDeepAttributes(out)
        
        return out
        
    
    @staticmethod
    def __addDeepAttributes(dic):
        for key, value in dic.items() :
            if hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                dic.update((key + '__' + k, val) for k, val in deep_items)

            elif isinstance(value, list) and hasattr(value[0], 'get_params'):
                for idx,item in enumerate(value) :
                    deep_items = item.get_params().items()
                    dic.update((key + '_' + str(idx) + '_' + k, val) for k, val in deep_items)
            
            return dic
        
    @staticmethod
    def distributionsType(distributions):
        
        if not (isinstance(distributions, list) or isinstance(distributions, np.ndarray)) : 
            raise ValueError('arg list is expected')
        if not all(isinstance(d, DistributionBase) for d in distributions):
            raise ValueError('a list of distribution is expected')

        if  all(d.dtype_ == distributions[0].dtype_ for d in distributions) :
            return distributions[0].dtype_
        else :
            return distType.UNDEFINED

        
    
    
    def elementWiseDecorator(self, fn):
        """ Decorate the pdf/pmf/cdf implied methods to perform an element wise evaluation.

        Parameters
        ----------
        f: The pdf/pmf/cdf function to decorate
        
        Returns
        -------
        Decorated function
        
        Note
         -------
        The decorator loops through the distributions and only evaluate for each distribution 
        the corresponding samples (according to the element-wise rule). The results are then aggregated back into a result list
        
        The iterative subseting of the vectorized distribution is made by simply modifying iteratively within the loop the 'cached_index'. 
        The implied evaluation methods will then automatically call the adequate distribution parameters (through the get_cached_param() method)

        """

        def wrapper(X, *args):

            n_ = utils.dim(X)

            if(n_ == 1):
                return fn(X)
                
            result = [0]*n_ 
            step = min(self.vectorSize(), n_)
            
            for index in range(step) :
                self.cached_index_ = index
                s = slice(index, n_ , step)
                at = X[s]
                result[s] = fn(at)
                
            self.reset()
            
            return result

        return wrapper
    


    def __getitem__(self, index  = None):
            """Returns a subset of the distribution object
           
            Parameters
            ----------
            - slice indexing, mode (optional)
            - mode only (in which full subset is returned)
            
            Returns
            -------
            ``skpro distribution object`
            """

            # parse key
            if isinstance(index, str) or index == None:
                selection = slice(None)
            else:
                selection = index

            # convert index to slice for consistent usage
            if isinstance(selection, int):
                if selection >= len(self):
                    raise IndexError('Selection is out of bounds')

                selection = slice(selection, selection)

            # check for out of bounds subsets
            if isinstance(index, slice) and index.stop > len(self):
                raise IndexError('Selection is out of bounds')

            # create subset replication
            replication = self.__class__(**self.__getitemParams(index))
            replication.reset()

            return replication



    def __len__(self):
        return self.vectorSize()
    
    
    def __eq__(self, other): 
        if not isinstance(other, self.__class__):
            raise ValueError('cannot compare different distribution')

        for key, value in self.get_params(deep = False).items() :
            if not (hasattr(other, key) and getattr(other, key, None) == value) :
                return False
        return True
        


    # interface methods
    def pdf(self, X):
        """ Main interface the pdf method. 
          
           Call the 'pdf_imp' implied method inherited from 'skpro.distributions.DPQRMixin'
           - Distribution must be of MIXED or CONTINUOUS type.
           - X is tested for being in the range of the distribution support. 
           - A elementWise decorator is called if the mode is set to 'ELEMENT_WISE'

        Parameters
         ----------
         X : array-like, shape = (n_samples, d_features)
            Test samples
            
        Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, m_distribution_size) in [BATCH-MODE]
                    (n_samples) in [ELEMENT-WISE-MODE]
        """
        
        
        if(self.dtype_ in [distType.DISCRETE, distType.UNDEFINED]):
            raise ValueError('pdf function not permitted for non continuous distribution')
            
        if(not self.support().inSupport(X)):
            raise ValueError('X is outside permitted support')

        if(self.mode_ is Mode.ELEMENT_WISE):
            func = self.elementWiseDecorator(self.pdf_imp)
            return func(X)
        
        return self.pdf_imp(X)


    def pmf(self, X):
        """ Main interface the pmf method. 
          
           Call the 'pmf_imp' implied method inherited from 'skpro.distributions.DPQRMixin'
           - Distribution must be of MIXED or DISCRETE type.
           - X is tested for being in the range of the distribution support. 
           - A elementWise decorator is called if the mode is set to 'ELEMENT_WISE'

         Parameters
         ----------
         X : array-like, shape = (n_samples, d_features)
            Test samples
            
         Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, m_distribution_size) in [BATCH-MODE]
                    (n_samples) in [ELEMENT-WISE-MODE]
        """
        
        if(self.dtype_ in [distType.CONTINUOUS, distType.UNDEFINED]):
            raise ValueError('pmf function not permitted for non continuous distribution')
            
        if(not self.support().inSupport(X)):
            raise ValueError('X is outside permitted support')
            
        if(self.mode_ is Mode.ELEMENT_WISE):
            func = self.elementWiseDecorator(self.pmf_imp)
            return func(X)
            
        return self.pmf_imp(X)


    def cdf(self, X):
        """ Main interface the cdf method. 
          
           Call the 'cdf_imp' implied method inherited from 'skpro.distributions.DPQRMixin'
           - X is tested for being in the range of the distribution support. 
           - A elementWise decorator is called if the mode is set to 'ELEMENT_WISE'

        Parameters
         ----------
         X : array-like, shape = (n_samples, d_features)
            Test samples
            
        Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, m_distribution_size) in [BATCH-MODE]
                    (n_samples) in [ELEMENT-WISE-MODE]
        """
        
        if(self.mode_ is Mode.ELEMENT_WISE):
            func = self.elementWiseDecorator(self.cdf_imp)
            return func(X)

        return self.cdf_imp(X)
    
    
    def squared_norm(self):
        """ Main interface the squared norm method. 
        Call the 'squared_norm_imp()' implied method inherited from 'skpro.distributions.DPQRMixin'

            
        Returns
         -------
         pdf output : ndarray of float
            shape = (n_samples, m_distribution_size) in [BATCH-MODE]
                    (n_samples) in [ELEMENT-WISE-MODE]
        """

        return self.squared_norm_imp()

        
        
    


