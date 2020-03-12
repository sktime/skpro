import numpy as np
from sklearn.base import BaseEstimator


class ClippedEstimator(BaseEstimator):
    """ clipped estimator

    Wrapping estimator that replaces predictions of the wrapped
    estimator that fall below a specified minimum threshold
    with the threshold itself.

    Parameters
    ----------
    estimator: subclass of sklearn.base.BaseEstimator
        Estimator which predicts shall be bounded by minimum threshold
    minimum: float/int
        Minimum boundary for the estimator's predictions. If relative=True
        the minimum represent a percentage value
    relative: bool
        If true, minimum will be regarded as percentage value
        and the cut-off threshold will be determined dynamically
        during fitting as ``threshold = minimum * std(y)``

    Properties
    ----------
    estimator : subclass of sklearn.base.BaseEstimator
        Wrapped estimator
    minimum : float
        Minimum threshold
    relative: bool
        If minimum is relative with regard to label variance
    """

    def __init__(self, clipped_estimator, minimum=0.3, relative=False):
        self.clipped_estimator = clipped_estimator
        self.minimum = minimum
        self.relative = relative

    def fit(self, X, y):
        self.clipped_estimator.fit(X, y)

        # Compute absolute minimum
        if self.relative:
            self.minimum *= np.std(y)

        return self

    def predict(self, X):
        # Apply cut-off
        prediction = np.clip(a = self.clipped_estimator.predict(X), a_max = None, a_min = self.minimum)
        return prediction


