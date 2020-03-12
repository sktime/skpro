import abc

from sklearn.base import BaseEstimator
import skpro.metrics.classical_loss as loss

from skpro.estimators.clipped import ClippedEstimator


class BaseResidualEstimator(BaseEstimator, metaclass=abc.ABCMeta):
    """ Base abstract for Residual estimators
    Used to predicts residuals part of a probabilistic parametric estimator.
    """
    
    def __init__(self) : 
        self.mean_estimator = None         
        self.is_linked = False
    
    
    def  __call__(self, mean_estimator):
        """
        must be called externally before fitting
        To attach [this] residual estimator to an estimator for the mean
    
        Parameters
        ----------
        mean_estimator: any sklearn BaseEstimator object (already fitted)
             use as a estimator for the mean 
        """

        self.mean_estimator = mean_estimator
        self.is_linked = True
    
    def reset(self):
        self.mean_estimator = None
        self.is_linked = False




class ResidualEstimator(BaseResidualEstimator):
    """ Concrete Residual estimator
    
    Parameters
        ----------
        estimator: any sklearn BaseEstimator object
             use as a base estimator for the dispersion
        
        loss_function : Classical Loss function object
             use as loss function during fitting

        Notes
        -----
        A scikit-learn estimator for the mean must first attached (before fitting) via the accept() method
        ELSE an error will be raised during fitting.
        

    """

    def __init__(self, base_estimator, loss_function = loss.SquaredError()
                 ) :      
        self.base_estimator = base_estimator
        self.loss_function = loss_function
        super().__init__()

    
    def fit(self, X, y, sample_weight):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.
            
        sample_weight : array-like, shape (n_samples,)
            Vector of weight to be eventually applied to the samples during trainning

        Returns
        -------
        self : object
        """
        
        if not self.is_linked: raise ValueError("mean estimator not linked to a mean_estimator")
        residuals = self.loss_function(self.mean_estimator.predict(X), y)
        self.base_estimator.fit(X, residuals)

        return self
    

    def predict(self, X):
        """returns a variance estimate

        Parameters
         ----------
          X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
            
        Returns
         ----------
         Array-like, shape(n_samples,)
         """
        
        return self.base_estimator.predict(X)

    
    
class ClippedResidualEstimator(ResidualEstimator):
    """ Concrete Clipped  Residual estimator
    
    Parameters
        ----------
        estimator: any sklearn BaseEstimator object
             use as a base estimator for the dispersion
        
        loss_function : Classical Loss function object
             use as loss function during fitting
  
        minWrapActive : Boolean
             specify is a minWrap correction should be applied to the prediction
             essentially acting as a floor to the predicted values

        minWrapValue : float
             specify the value of the floor to be used if the
             minWrapActive argument is set to TRUE

        Notes
        -----
        A scikit-learn estimator for the mean must first attached (before fitting) via the accept() method
        ELSE an error will be raised during fitting.
        

    """

    def __init__(self, clipped_base_estimator, loss_function = loss.SquaredError(), minimum = 0.0, relative = False):
        self.minimum = minimum
        self.clipped_base_estimator = clipped_base_estimator
        
        estimator =  ClippedEstimator(clipped_base_estimator, minimum, relative)
        super().__init__(estimator, loss_function)
        
     


