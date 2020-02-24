import numpy as np
from scipy.stats import norm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

from skpro.workflow.manager import DataManager
from skpro.baselines.classical_baselines import ClassicalBaseline

from skpro.estimators.parametric import ParametricEstimator
from skpro.estimators.residuals import ResidualEstimator

from skpro.metrics.proba_loss_cont import LogLossClipped
from skpro.metrics.proba_scorer import ProbabilisticScorer


def test_dummy_prediction():
   
    data = DataManager('boston')

    model = ClassicalBaseline()
    model.fit(data.X_train, data.y_train)
    y_pred = model.predict_proba(data.X_test)

    mu = np.mean(data.y_train)
    sigma = np.std(data.y_train)

    # is the dummy prediction working?
    assert (y_pred.point() == np.ones((len(data.X_test))) * mu).all()
    assert (y_pred.std() == np.ones((len(data.X_test))) * sigma).all()

    # does subsetting work?
    assert len(y_pred[1:2]) == 2
    assert y_pred[1:3].point() == 3 * [mu]

    # pdf, cdf?
    x = np.random.randint(0, 10)
    i = np.random.randint(0, len(data.X_test) - 1)

    rtol = 1e-05
    atol = 1e-08

    assert np.allclose(y_pred[i].pdf(x), norm.pdf(x, mu, sigma), rtol, atol)
    assert np.allclose(y_pred[i].cdf(x), norm.cdf(x, mu, sigma), rtol, atol)


def test_static_linear_mean_prediction():
    
    rtol = 1e-05
    atol = 1e-08
    
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= False)

    model = ParametricEstimator(LinearRegression(), LinearRegression())
    distribution = model.fit(X_train, y_train).predict_proba(X_test)
    prediction = distribution.mean()

    static_pred = [24.88963777, 23.72141085, 29.36499868, 12.12238621, 21.44382254, 19.2834443,
                   20.49647539, 21.36099298, 18.8967118,  19.9280658,   5.12703513, 16.3867396,
                   17.07776485,  5.59375659, 39.99636726, 32.49654668, 22.45798809, 36.85192327,
                   30.86401089, 23.15140009, 24.77495789, 24.67187756, 20.59543752, 30.35369168,
                   22.41940736, 10.23266565, 17.64816865, 18.27419652, 35.53362541, 20.96084724,
                   18.30413012, 17.79262072, 19.96561663, 24.06127231, 29.10204874, 19.27774123,
                   11.15536648, 24.57560579, 17.5862644,  15.49454112, 26.20577527, 20.86304693,
                   22.31460516, 15.60710156, 23.00363104, 25.17247952, 20.11459464, 22.90256276,
                   10.0380507 , 24.28515123, 20.94127711, 17.35258791, 24.52235405, 29.95143046,
                   13.42695877, 21.72673066, 20.7897053 , 15.49668805, 13.98982601, 22.18377874,
                   17.73047814, 21.58869165, 32.90522136, 31.11235671, 17.73252635, 32.76358681,
                   18.7124637,  19.78693475, 19.02958927, 22.89825374, 22.96041622, 24.02555703,
                   30.72859326, 28.83142691, 25.89957059,  5.23251817, 36.72183202, 23.77267249,
                   27.26856352, 19.29492159, 28.62304496, 19.17978838, 18.97185995, 37.82397662,
                   39.22012647, 23.71261106, 24.93076217, 15.88545417, 26.09845751, 16.68819641,
                   15.83515991, 13.10775597, 24.71583588, 31.25165267, 22.16640989, 20.25087212,
                   0.59025319, 25.44217132, 15.57178328, 17.93719475, 25.30588844, 22.3732326]

    assert(np.allclose(prediction , static_pred, rtol, atol))
    

   
 
def test_residual_prediction():

     data = DataManager('boston')
     
     residualModel = ResidualEstimator(estimator = LinearRegression(), minWrapActive = True, minWrapValue = 2**4)
     model = ParametricEstimator(LinearRegression(), residualModel)
     model.fit(data.X_train, data.y_train)
  
     baseline =  ClassicalBaseline(LinearRegression()).fit(data.X_train, data.y_train)

     score = ProbabilisticScorer(LogLossClipped())
     baseline_loss = score(baseline, data.X_test, data.y_test)
     y_pred_loss = score(model, data.X_test, data.y_test)

     assert baseline_loss > y_pred_loss


    
    
if __name__ == "__main__":

    test_dummy_prediction()
    test_static_linear_mean_prediction()
    test_residual_prediction()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    