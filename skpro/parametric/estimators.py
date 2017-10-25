import numpy as np
from sklearn.base import BaseEstimator


class Minimum(BaseEstimator):
    """ Minimum estimator

    Wrapping estimator that replaces predictions of the wrapped
    estimator that fall below a specified minimum threshold
    with the threshold itself.

    Parameters
    ----------
    estimator: subclass of sklearn.base.BaseEstimator
        Estimator which predicts shall be bounded by minimum threshold
    minimum: float
        Minimum boundary for the estimator's predictions

    Properties
    ----------
    estimator : subclass of sklearn.base.BaseEstimator
        Wrapped estimator
    minimum : float
        Minimum threshold

    """

    def __init__(self, estimator, minimum=3):
        self.estimator = estimator
        self.minimum = minimum

    def fit(self, X, y):
        if getattr(self, 'estimator', False):
            self.estimator.estimator = self.estimator
        self.estimator.fit(X, y)

        return self

    def predict(self, X):
        prediction = self.estimator.predict(X)
        prediction[prediction < self.minimum] = self.minimum

        return prediction

    def __str__(self, describer=str):
        return 'Min(' + describer(self.estimator) + ', min=' + str(self.minimum) + ')'

    def __repr__(self):
        return self.__str__(repr)


class Constant(BaseEstimator):
    """ Constant estimator

    Predicts predefinied constant

    Parameters
    ----------
    constant: float | callable(X, y) | string: 'mean(y)', 'std(y)'
        Specifies the constant. A callable receives the training data during
        fit and should return a constant value. The string options provide
        a shortcut for mean/std extraction from the features.
    name: string (optional)
        Optional description of the constant for the estimator string
        representation. Defaults to str(constant).
    """

    def __init__(self, constant=None, name=None):
        self.constant = constant
        self.name = name

    def fit(self, X, y):
        # evaluate callables
        if callable(self.constant):
            self.constant = self.constant(X, y)

        # resolve str shorthands
        if isinstance(self.constant, str):
            self.name = self.constant
            if self.constant == 'mean(y)':
                self.constant = np.mean(y)
            elif self.constant == 'std(y)':
                self.constant = np.std(y)
            else:
                raise ValueError(self.constant + ' is not a valid function')

        # if no constant was provided we use the y value
        if self.constant is None:
            self.constant = y

        return self

    def predict(self, X):
        return np.ones((X.shape[0],)) * self.constant

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.name is None:
            return 'C(' + str(self.constant) + ')'
        else:
            return 'C(' + self.name + ')'
