import numpy as np
from sklearn.base import BaseEstimator


def identity(y_pred):
    return y_pred


def squared_error_ft(y, y_pred):
    return (y - y_pred) ** 2


def squared_error_pt(y_pred):
    return np.sqrt(np.abs(y_pred))


def abs_error_ft(y, y_pred):
    return np.abs(y - y_pred)


abs_error_pt = identity


def log_error_ft(y, y_pred):
    residuals = np.abs(y - y_pred)
    residuals[residuals <= 1] = 1
    return np.log(residuals)


def log_error_pt(y_pred):
    return np.exp(y_pred)


class ResidualEstimator(BaseEstimator):
    """ Residual estimator

    Predicts residuals of an estimator using a scikit-learn estimator.

    TODO: expand documentation
    """

    def __init__(self, residual_estimator, base_estimator='point', fit_transform='squared_error', predict_transform=None, filter_zero_variance=False):
        self.residual_estimator = residual_estimator
        self.base_estimator = base_estimator
        self.fit_transform = fit_transform
        self.fit_transform_ = fit_transform
        self.predict_transform = predict_transform
        self.predict_transform_ = predict_transform
        self.filter_zero_variance = filter_zero_variance

    def _resolve_transformer(self, tf, suffix):
        if isinstance(tf, str):
            try:
                return globals()[tf + suffix]
            except:
                raise ValueError(tf + ' is not a valid transformer')
        else:
            return None

    def _resolve_transformers(self):
        self.fit_transform_ = self._resolve_transformer(self.fit_transform, '_ft')
        self.predict_transform_ = self._resolve_transformer(self.predict_transform, '_pt')
        if self.predict_transform_ is None:
            self.predict_transform_ = self._resolve_transformer(self.fit_transform, '_pt')

    def fit(self, X, y):
        self._resolve_transformers()

        y_pred = self.estimator.estimators.predict(self.base_estimator, X)

        # protect against 0 variance
        if self.filter_zero_variance:
            clean = (y - y_pred != 0)
            if np.any(clean):
                y = y[clean]
                X = X[clean]

        # retrieve fitting vars
        y_ = self.fit_transform_(y, y_pred)
        self.residual_estimator.fit(X, y_)

        return self

    def predict(self, X):
        y_pred = self.residual_estimator.predict(X)

        # apply transformation
        y_pred_ = self.predict_transform_(y_pred)

        return y_pred_

    def __str__(self):
        return 'RE(' + str(self.base_estimator) + ', ' \
               + str(self.residual_estimator) + ', ' + str(self.fit_transform) + ')'

    def __repr__(self):
        return 'ResidualEstimator(' + str(self.base_estimator) + ', ' \
               + repr(self.residual_estimator) + ', ' + repr(self.fit_transform) + ')'


