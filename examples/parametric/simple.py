from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets.base import load_boston
from sklearn.model_selection import train_test_split

from skpro.parametric import ParametricEstimator
from skpro.parametric.estimators import Constant
from skpro.metrics import log_loss

# Define the parametric model
model = ParametricEstimator(
    point=RandomForestRegressor(),
    std=Constant('std(y)'),
    shape='norm'
)

# Train and predict on boston housing data
X, y = load_boston(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
y_pred = model.fit(X_train, y_train).predict(X_test)

# Obtain the loss
loss = log_loss(y_test, y_pred, sample=True, return_std=True)
print('Loss: %f+-%f' % loss)

# Plot the performance
import sys
sys.path.append('../')
import skpro.examples.utils
utils.plot_performance(y_test, y_pred)