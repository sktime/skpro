import numpy as np
from sklearn.base import BaseEstimator


class Minimum(BaseEstimator):

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

    def __init__(self, constant=None, name=None):
        """

        :param constant: Constant value, or callable(X, y) returning a constant, or str 'mean(y)'|'std(y)'
        :param name: Optional description
        """
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

