"""Implements predictive target calibration for probabilistic regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["arnavk23"]
__all__ = ["DistrPredictiveCalibration"]

from sklearn.base import BaseEstimator

from skpro.regression.base import BaseProbaRegressor


class _IdentityCalibrator(BaseEstimator):
    """Identity calibrator used in estimator checks for test instance creation.

    Kept at module scope so sklearn cloning and serialization remain robust in tests.
    """

    def fit(self, y_true, y_pred):
        return self

    def transform(self, y_pred):
        return y_pred


class _ScaleOnlyCalibrator(BaseEstimator):
    """Simple deterministic calibrator for estimator check parametrization."""

    def __init__(self, scale=1.1):
        self.scale = scale

    def fit(self, y_true, y_pred):
        return self

    def transform(self, y_pred):
        return y_pred


class DistrPredictiveCalibration(BaseProbaRegressor):
    """DistrPredictiveCalibration pipeline for predictive target calibration.

    Wraps a probabilistic regressor and applies a calibration method
    to its predicted distributions.

    Parameters
    ----------
    regressor : BaseProbaRegressor
        The probabilistic regressor to wrap.
    calibrator : object
        The calibration method to apply to predicted distributions.
        Must implement fit(y_true, y_pred) and transform(y_pred).

    Examples
    --------
    >>> from skpro.regression.compose import DistrPredictiveCalibration
    >>> from skpro.regression.residual import ResidualDouble
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> import pandas as pd
    >>> # Dummy calibrator for demonstration
    >>> from sklearn.base import BaseEstimator, TransformerMixin
    >>> class DummyCalibrator(BaseEstimator, TransformerMixin):
    ...     def fit(self, y_true, y_pred):
    ...         return self
    ...     def transform(self, y_pred):
    ...         return y_pred
    >>> # Load data
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> y = pd.DataFrame(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> reg = ResidualDouble.create_test_instance()
    >>> cal = DummyCalibrator()
    >>> calreg = DistrPredictiveCalibration(regressor=reg, calibrator=cal)
    >>> calreg.fit(X_train, y_train)
    DistrPredictiveCalibration(...)
    >>> y_pred = calreg.predict(X_test)
    >>> y_pred_proba = calreg.predict_proba(X_test)
    >>> # Note: Calibrator must accept and return distribution objects
    >>> # as output from predict_proba.
    """

    _tags = {
        "capability:multioutput": True,
        "capability:missing": True,
    }

    def __init__(self, regressor, calibrator):
        self.regressor = regressor
        self.calibrator = calibrator
        super().__init__()

    def _fit(self, X, y, C=None):
        from sklearn.base import clone

        # Clone regressor and calibrator to avoid mutating input parameters
        self._fitted_regressor = clone(self.regressor)
        self._fitted_regressor.fit(X, y, C=C)
        self._fitted_calibrator = clone(self.calibrator)
        # Fit calibrator on training predictions
        y_pred = self._fitted_regressor.predict_proba(X)
        self._fitted_calibrator.fit(y, y_pred)
        return self

    def _predict(self, X):
        return self._fitted_regressor.predict(X)

    def _predict_quantiles(self, X, alpha):
        y_pred = self._fitted_regressor.predict_quantiles(X, alpha)
        return self._fitted_calibrator.transform(y_pred)

    def _predict_interval(self, X, coverage):
        y_pred = self._fitted_regressor.predict_interval(X, coverage)
        return self._fitted_calibrator.transform(y_pred)

    def _predict_var(self, X):
        y_pred = self._fitted_regressor.predict_var(X)
        return self._fitted_calibrator.transform(y_pred)

    def _predict_proba(self, X):
        y_pred = self._fitted_regressor.predict_proba(X)
        return self._fitted_calibrator.transform(y_pred)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter sets for automated tests.

        Uses explicit calibrators to exercise constructor and set/get param checks.
        """
        from skpro.regression.residual import ResidualDouble

        reg = ResidualDouble.create_test_instance()
        return [
            {"regressor": reg, "calibrator": _IdentityCalibrator()},
            {"regressor": reg, "calibrator": _ScaleOnlyCalibrator(scale=1.05)},
        ]
