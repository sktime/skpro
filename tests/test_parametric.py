import numpy as np

from skpro.workflow.manager import DataManager
from skpro.parametric import ParamtericEstimator


def test_baseline():
    data = DataManager('boston')

    model = ParamtericEstimator()

    y_pred = model.fit(data.X_train, data.y_train).predict(data.X_test)

    assert np.isclose(y_pred.point(), data.y_test, atol=np.max(data.y)).all()