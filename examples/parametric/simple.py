from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets.base import load_boston
from sklearn.model_selection import train_test_split

from skpro.parametric import ParamtericEstimator
from skpro.parametric.estimators import Constant
from skpro.metrics import log_loss

# Define the parametric model
model = ParamtericEstimator(
    point=RandomForestRegressor(),
    std=Constant('mean(y)')
)

# Train and predict on boston housing data
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_pred = model.fit(X_train, y_train).predict(X_test)

# Obtain the loss
loss = log_loss(y_pred, y_test, sample=True, return_std=True)
print('Loss: %f+-%f' % loss)

# Plot the performance

from matplotlib import pyplot
import numpy as np

fig, ax1 = pyplot.subplots()

ax1.plot(y_test, y_test, 'g-', label=u'Optimum')
sigma = np.std(y_pred) / np.sqrt(len(y_pred))
ax1.errorbar(y_test, y_pred, yerr=sigma, fmt='b.', label=u'Predictions', ecolor='r', elinewidth='0.5')
ax1.set_ylabel('Predicted $y_{pred}$')
ax1.set_xlabel('Correct label $y_{true}$')
ax1.legend(loc='best')
pyplot.title('Prediction performance plot')

losses = log_loss(y_pred, y_test, sample=False)
ax2 = ax1.twinx()
overall = "{0:.2f}".format(np.mean(losses)) + " +/- {0:.2f}".format(np.std(losses))
ax2.set_ylabel('Loss')
ax2.plot(y_test, losses, 'y_', label=u'Loss: ' + overall)
ax2.tick_params(colors='y')
ax2.legend(loc=0)


pyplot.show()