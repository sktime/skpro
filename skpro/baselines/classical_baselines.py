
from skpro.estimators.parametric import ParametricEstimator
from skpro.distributions.distribution_normal import NormalDistribution
from skpro.baselines.dummy_regressor import DummyRegressor




class ConstantUninformedBaseline(ParametricEstimator) :
    
    def __init__(self, constant = (42, 42), distribution = NormalDistribution(), copy_X=True):
        
        if(isinstance(constant, tuple)):
            cmean = constant[0]
            cvariance = constant[1]
        else:
            cmean = cvariance = constant

        super().__init__(
                 mean_estimator = DummyRegressor(strategy="constant", constant = cmean), 
                 dispersion_estimator = DummyRegressor(strategy="constant", constant = cvariance), 
                 distribution = distribution, 
                 residuals_strategy = False,
                 copy_X = copy_X
          )
        


class ClassicalBaseline(ParametricEstimator) :
    
    def __init__(self, mean_estimator = DummyRegressor('mean'), distribution = NormalDistribution(), residuals_strategy = False, copy_X=True):
       super().__init__(
               mean_estimator, 
               dispersion_estimator = DummyRegressor('variance'), 
               distribution = distribution,
               residuals_strategy  = residuals_strategy,
               copy_X = copy_X
              )
    
    