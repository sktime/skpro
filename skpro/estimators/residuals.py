import numpy as np
from sklearn.base import BaseEstimator
import skpro.metrics.classical_loss as loss


class ResidualEstimator(BaseEstimator):
    """ Residual estimator

    Predicts residuals of an estimator using a scikit-learn estimator.

    Read more in the :ref:`User Guide <parametric>`.
    """

    def __init__(self, estimator, loss_function = loss.SquaredError(),  minWrap = True, minWrapValue = 0.0) :      
        self.minWrap = minWrap
        self.minWrapValue = minWrapValue
        self.base_estimator = estimator
        self.loss_function = loss_function
        
        self.mean_estimator = None
        self.is_linked = False
        
        
    def linkToEstimator(self, mean_estimator):
        self.mean_estimator = mean_estimator
        self.is_linked = True
        
    def fit(self, X, y, sample_weight):
        
        if not self.is_linked:
              raise ValueError("mean estimator not linked")
              
        if not self.mean_estimator.is_fitted :
            self.mean_estimator.fit(X, y, sample_weight)
            
        residuals = self.loss_function(self.mean_estimator.predict(X), y)
        self.residual_estimator.fit(X, residuals)

        return self
    

    def predict(self, X):
        variancePrediction = self.estimator.predict(X)
        
        if self.minWrap == True :
            variancePrediction = np.clip(a = variancePrediction, a_max = None, a_min = self.minWrapValue)

        return variancePrediction

