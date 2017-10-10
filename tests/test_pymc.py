from skpro.workflow.manager import DataManager

from skpro.pymc import PyMC


def test_construct_estimator(self):
    data = DataManager('boston')

    model = PyMC()

    y_pred = model.fit(data.X_train, data.y_train).predict(data.X_test)







