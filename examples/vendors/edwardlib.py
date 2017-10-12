from skpro.metrics import log_loss

from skpro.vendors.edwardlib import Edwardlib, EdwardlibInterface
from skpro.workflow.manager import DataManager


# Plug the model definition into the PyMC interface

model = Edwardlib(
    model=EdwardlibInterface()
)


# Run prediction, print and plot the results

data = DataManager('boston')
y_pred = model.fit(data.X_train, data.y_train).predict(data.X_test)
print('Log loss: ', log_loss(data.y_test, y_pred, return_std=True))

from matplotlib import pyplot
pyplot.scatter(y_pred.point(), data.y_test)
pyplot.show()