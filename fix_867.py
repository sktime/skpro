# skpro/base/_base.py

import copy

class BaseEstimator:
    def __init__(self, **params):
        self.params = params

    def predict_proba(self, X):
        # Simulate a scenario where a parameter is mutated
        self.params['mutation'] = True
        return X

    def get_test_params(self):
        # Ensure that the test parameters include a case that triggers mutation
        return {
            'param1': 1,
            'param2': 2
        }

# skpro/benchmarking/tests/test_evaluate.py

import pytest
from skpro.base._base import BaseEstimator

def test_predict_proba_parameter_mutation():
    estimator = BaseEstimator(param1=1, param2=2)
    initial_params = copy.deepcopy(estimator.params)
    
    estimator.predict_proba([[1, 2, 3]])
    
    # Check if the parameter 'mutation' was added
    assert 'mutation' in estimator.params
    assert estimator.params['mutation'] is True
    
    # Ensure that other parameters remain unchanged
    for key, value in initial_params.items():
        if key != 'mutation':
            assert estimator.params[key] == value