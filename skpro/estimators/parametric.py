import scipy.sparse as sp

from sklearn.utils.validation import check_X_y

from skpro.estimators.base import ProbabilisticEstimator
from skpro.estimators.score import DefaultScoreMixin
from skpro.distributions.distribution_normal import Normal
from skpro.distributions.location_scale import LocationScaleMixin
from skpro.estimators.residuals import BaseResidualEstimator, ClippedResidualEstimator


class ParametricEstimator(ProbabilisticEstimator, DefaultScoreMixin):
    """
    Composite parametric prediction strategy.
    Uses classical estimators to predict the defining parameters of continuous distributions.
    
    Parameters
        ----------
        mean_estimator: any sklearn BaseEstimator object
             use as a standard estimator for the mean parameter of the probabilistic regression
        
        dispersion_estimator : ResidualEstimator object or any sklearn BaseEstimator object
             use as a standard estimator for the dispersion of the composite estimator
  
        distribution : skpro distribution object if 
             will be use to defined the class of the distribution class type to be used for prediction (through type(self.distribution)). 
             The composite estimator now is restricted to 
             univariate normal or laplace distribution (ELSE will raise an Error).
             
        residuals_strategy : boolean
             specify wether the estimator should perform a residual strategy for the dispersion estimation
             (i.e. first fit the mean estimate and then fit the dispersion as error relative to these mean estimates)

        copy_X : boolean
             
        Notes
        -----
        If 'residuals_strategy' is set to TRUE then the dispersion_estimator member 
        MUST be of type ResidualEstimator (i.e. skpro.estimators.residuals). 
        Thus : -> IF dispersion_estimator init argument is already of type ResidualEstimator the argument is simply passed as class member
               -> ELSE a new ResidualEstimator estimator is instanciated within the init method (with the non residual estimator as argument)  

    """
    
    def __init__(self, mean_estimator, dispersion_estimator, 
                 distribution = Normal(), residuals_strategy = True, copy_X=True):
        
        if isinstance(dispersion_estimator, BaseResidualEstimator) :
            self.dispersion_estimator = dispersion_estimator
            residuals_strategy = True
            
        elif(residuals_strategy):
            self.dispersion_estimator =  ClippedResidualEstimator(dispersion_estimator)
        else : self.dispersion_estimator = dispersion_estimator

        self.mean_estimator = mean_estimator
        self.distribution = distribution
        self.copy_X = copy_X
        self.residuals_strategy = residuals_strategy
        
        if(self.residuals_strategy):
            if not isinstance(self.dispersion_estimator, BaseResidualEstimator):
                raise ValueError("dispersion estimator must be of BaseResidualEstimator type for residuals stratigies")

            self.dispersion_estimator(self.mean_estimator)
        
    
    def fit(self, X, y, sample_weight=None):
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
        
        Notes
        -------
        As a fitting procedure :
            - the mean estimator is first fitted. 
            - If the residual_strategy type is active : The fitted mean_estimator is passed 
              to dispersion_estimator on which is dependent (through its accept() method)
            - the dispersion_estimator is then fitted as well
        """

        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         y_numeric=True, multi_output=True)
        
        if self.copy_X :
            
            if sp.issparse(X):
                X = X.copy()
            else:
                X = X.copy(order='K')

        self.mean_estimator.fit(X, y, sample_weight)
        self.dispersion_estimator.fit(X, y, sample_weight)
        
        return self
    
    
    def predict_proba(self, X):
        """probabilistic prediction method. Return a (vectorized) predicted distribution object.
         The composite estimator is restricted to univariate normal or laplace distribution (ELSE will raise an Error).

        Parameters
         ----------
          X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
            
        Returns
         ----------
         skpro distribution object, (eventually vectorized)
         
         Notes
         ----------
         1. It essentially calling the 'fitted dipersion_estimator' to estimate a vector of distribution variance from features
         2. Then convert it into scale parameter using the "varianceToScale" method 
            that MUST be implemented in the candidate distribution class  (ELSE will raise an Error).
        
            
        """
        
        if not isinstance(self.distribution, LocationScaleMixin):
            raise ValueError("Unknown strategy type : expected a location_scale distribution")

        if self.copy_X :
            if sp.issparse(X):
                X = X.copy()
            else:
                X = X.copy(order='K')
        
        dispersionPrediction = self.dispersion_estimator.predict(X)
        
        #if(not hasattr(self.distribution, "varianceToScale")):
                #raise ValueError("No 'varianceToScale' class.method implemented for : %s distribution"
                            # % (self.distribution.name()))
        
        scale = type(self.distribution).varianceToScale(dispersionPrediction)

        return type(self.distribution)(
                self.mean_estimator.predict(X),
                scale
                )
