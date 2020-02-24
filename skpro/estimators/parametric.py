import scipy.sparse as sp

from sklearn.utils.validation import check_X_y

from skpro.distributions.distribution_normal import NormalDistribution
from skpro.estimators.base import ProbabilisticEstimator
from skpro.estimators.residuals import ResidualEstimator


class ParametricEstimator(ProbabilisticEstimator):
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
                 distribution = NormalDistribution(), residuals_strategy = True, copy_X=True):
        
        if isinstance(dispersion_estimator, ResidualEstimator) :
            self.dispersion_estimator_ = dispersion_estimator
            residuals_strategy = True
            
        elif(residuals_strategy):
            self.dispersion_estimator_ =  ResidualEstimator(dispersion_estimator)
            
        else :
            self.dispersion_estimator_ = dispersion_estimator

        self.mean_estimator_ = mean_estimator
        self.distribution_ = distribution
        self.copy_X_ = copy_X
        self.residuals_strategy_ = residuals_strategy
        
    
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
        
        if self.copy_X_ :
            if sp.issparse(X):
                X = X.copy()
            else:
                X = X.copy(order='K')

        self.mean_estimator_.fit(X, y, sample_weight)
        
        if(self.residuals_strategy_):
            self.dispersion_estimator_.accept(self.mean_estimator_)
            
        self.dispersion_estimator_.fit(X, y, sample_weight)
        
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

        allowed_distribution = ("normal", "laplace")
        if self.distribution_.name() not in allowed_distribution:
            raise ValueError("Unknown strategy type: %s, expected one of %s."
                             % (self.distribution_.name(), allowed_distribution))

        if self.copy_X_ :
            if sp.issparse(X):
                X = X.copy()
            else:
                X = X.copy(order='K')
        
        dispersionPrediction = self.dispersion_estimator_.predict(X)
        
        if(not hasattr( self.distribution_, "varianceToScale")):
                raise ValueError("No 'varianceToScale' class.method implemented for : %s distribution"
                             % (self.distribution_.name()))
        
        scale = type(self.distribution_).varianceToScale(dispersionPrediction)

        return type(self.distribution_)(
                self.mean_estimator_.predict(X),
                scale
                )
    

    def __repr__(self):
        return self.__str__(repr)

