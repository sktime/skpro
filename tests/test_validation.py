import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

from skpro.workflow.manager import DataManager
from skpro.estimators.parametric import ParametricEstimator
from skpro.estimators.residuals import ClippedResidualEstimator
from skpro.baselines.classical_baselines import ClassicalBaseline
from skpro.metrics.proba_loss_cont import LogLossClipped
from skpro.metrics.proba_scorer import ProbabilisticScorer
from skpro.cross_validation import ModelManager


def test_simple_score() :
    
     data = DataManager('boston', split=0.2, random_state=False)

     residualModel = ClippedResidualEstimator(clipped_base_estimator = LinearRegression(), minimum = 2**4)
     model = ParametricEstimator(LinearRegression(), residualModel)
     model.fit(data.X_train, data.y_train)
     
     rtol = 1e-05
     atol = 1e-08

     score = ProbabilisticScorer(LogLossClipped())
     target = 307.748721
     
     #assess scorer
     assert(np.allclose(score(model, data.X_test, data.y_test), target, rtol, atol))
     
     #assess default score method (i.e. log loss for parametric estimator)
     assert(np.allclose(model.score(data.X_test, data.y_test), target, rtol, atol))
    
    
    
def test_cross_validation_with_callable():
    data = DataManager('boston', split=0.2, random_state=False)
    
    rtol = 1e-05
    atol = 1e-08

    residualModel = ClippedResidualEstimator(clipped_base_estimator = LinearRegression(), minimum = 2**4)
    model = ParametricEstimator(LinearRegression(), residualModel)
    scorer = ProbabilisticScorer(LogLossClipped())
     
    cv = cross_val_score(
            model,
            data.X,
            data.y,
            scoring=scorer,
            cv=10
        )
    out = [132.78724027, 139.49742008, 142.94813629, 163.23139818, 168.65800343, 
           149.40938576, 131.05486794, 316.82750215, 164.05491674, 150.73567347]
    
    assert(np.allclose(cv, out, rtol, atol))
    
def test_cross_validation_default():
    data = DataManager('boston', split=0.2, random_state=False)
    
    rtol = 1e-05
    atol = 1e-08

    residualModel = ClippedResidualEstimator(clipped_base_estimator = LinearRegression(), minimum = 2**4)
    model = ParametricEstimator(LinearRegression(), residualModel)

    cv = cross_val_score(
            model,
            data.X,
            data.y,
            cv=10
        )
    out = [132.78724027, 139.49742008, 142.94813629, 163.23139818, 168.65800343, 
           149.40938576, 131.05486794, 316.82750215, 164.05491674, 150.73567347]
    
    assert(np.allclose(cv, out, rtol, atol))
    
    
def test_model_manager():
    
    scoring = ProbabilisticScorer(LogLossClipped())
    data = DataManager('boston', split=0.2, random_state=False)
    
    rtol = 1e-05
    atol = 1e-08
    
    baseline = ClassicalBaseline(LinearRegression())
    r1 = ClippedResidualEstimator(clipped_base_estimator = LinearRegression(), minimum = 2**4)
    linear_m = ParametricEstimator(LinearRegression(), r1)
   
    r2 = ClippedResidualEstimator(clipped_base_estimator = LinearRegression(), minimum = 2**4)
    tree_m = ParametricEstimator(DecisionTreeRegressor(), r2)

    cmp = ModelManager(data)
    cmp.register('classical', baseline)
    cmp.register('linear', linear_m)
    cmp.register('DT', tree_m)
    cmp.fit()
    
   
    score = cmp.score(scoring)
    assert(score.shape == (3,2))
    assert(np.allclose(score.loc['linear', 'test_score'], 307.748721, rtol, atol))
    assert(np.allclose(score.loc['linear', 'train_score'], 1143.717657, rtol, atol))
    assert(np.allclose(score.loc['classical', 'test_score'], 340.428615, rtol, atol))
    assert(np.allclose(score.loc['classical', 'train_score'], 1314.859241, rtol, atol))

    cv = cmp.cross_val_score(scoring = scoring,  n_splits=10)
    assert(cv.shape == (3,2))
    
    
    
if __name__ == "__main__":
    test_simple_score()
    test_cross_validation_with_callable()
    test_cross_validation_default()
    test_model_manager()

    
    