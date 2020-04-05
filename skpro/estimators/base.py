
import abc
from sklearn.base import BaseEstimator


class ProbabilisticEstimator(BaseEstimator, metaclass=abc.ABCMeta):
    """ Abstract base class for probabilistic prediction models

    Notes
    -----
    All probabilistic estimators should specify all the parameters
    that can be set at the class level in their ``__init__``
    as explicit keyword arguments (no ``*args`` or ``**kwargs``).
    """


    def __init__(self):
            if callable(getattr(self, '_init', None)):
                self._init()
                

    def name(self):
        return self.__class__.__name__

    def __str__(self):
        return '%s()' % self.__class__.__name__

    def __repr__(self):
        return '%s()' % self.__class__.__name__
    
    
    def predict(self, X):
        """classical point prediction method. 
        Return the point estimate associated with the distribution of the
        probabilistic prediction.
    
         Parameters
         ----------
          X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
             

         Returns
         ----------
         array-like, shape =  (n_samples)
            vector of predictions
         """

        distribution = self.predict_proba(X = X)
        return distribution.point()
    

    @abc.abstractmethod
    def predict_proba(self, X):
        """probabilistic prediction method. Return a (vectorized) predicted distribution object
        [Abstract method] : must be instantiated in the concrete sub

        Parameters
         ----------
          X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        
        Returns
         ----------
         skpro distribution object, (eventually vectorized)
            
        """
        raise NotImplementedError()



    @abc.abstractmethod
    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.
        [Abstract method] : must be instantiated in the concrete sub

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        Returns
        -------
        self : object
        """
        raise NotImplementedError()

        return self
