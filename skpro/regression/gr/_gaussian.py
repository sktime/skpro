"""Implements Gaussian regression using sklearn's LinearRegression."""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["sanjayk0508"]

import numpy as np
from scipy.stats import norm

from skpro.utils.validation import check_X_y, check_is_fitted, check_array

from skpro.regression.adapters.sklearn import SklearnProbaReg
from skpro.regression.base.adapters import _DelegateWithFittedParamForwarding


class GaussianRegressor(_DelegateWithFittedParamForwarding):
    """Gaussian Regressor, adapter to sklearn's LinearRegression.

    Fit a Gaussian regression model using LinearRegression from sklearn.
    This regressor assumes that the response variable follows a Gaussian
    distribution.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (i.e., data is expected to be centered).

    normalize : bool, default=False
        This parameter is ignored when `fit_intercept` is set to False.
        If True, the regressors X will be normalized before regression
        by subtracting the mean and dividing by the l2-norm.
        If you wish to standardize, please use `StandardScaler`
        before calling `fit` on the regressor.

    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model.

    intercept_ : float
        Independent term in the linear model. Set to 0.0 if
        `fit_intercept = False`.

    residuals_ : array-like of shape (n_samples,)
        Residuals of the training data.
    """

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True):
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X

        from sklearn.linear_model import LinearRegression

        skl_estimator = LinearRegression(
            fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X
        )

        skpro_est = SklearnProbaReg(skl_estimator)
        self._estimator = skpro_est.clone()

        super().__init__()

    FITTED_PARAMS_TO_FORWARD = ["coef_", "intercept_", "residuals_"]

    def __repr__(self):
        return (
            f"GaussianRegressor("
            f"fit_intercept={self.fit_intercept}, "
            f"normalize={self.normalize}, "
            f"copy_X={self.copy_X}"
            ")"
        )

    def __str__(self):
        return self.__repr__()

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        return self._estimator.get_params(deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Parameters
        ----------
        **params : dict
            Estimator parameters.
        """
        self._estimator.set_params(**params)

    def _fit(self, X, y):
        """Fit the regressor to training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : GaussianRegressor
            Fitted regressor.
        """
        X, y = check_X_y(X, y)
        check_is_fitted(self, ["_estimator"])

        self._estimator.fit(X, y)
        self.residuals_ = y - self.predict(X)
        return self

    def score(self, X, y):
        """
        Return the negative log-likelihood of the Gaussian distribution.

        Note: Typically, 'score' functions return a value between 0-1 indicating
        the performance of the model (1 being the best). However, in this case,
        the 'score' method returns the negative log-likelihood of the Gaussian
        distribution for the residuals, so lower values indicate better model
        performance.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            True labels for X.

        Returns
        -------
        score : float
            Negative log-likelihood of the Gaussian distribution.
        """
        check_is_fitted(self, ["_estimator"])

        residuals = y - self.predict(X)
        return -norm.logpdf(residuals).sum()

    def predict_proba(self, X):
        """
        Returns predicted means and standard deviations for each sample.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
        Each row corresponds to a sample in `X`. The first column represents
        the mean of the distribution, and the second column represents the
        standard deviation.
        """
        check_is_fitted(self, ["_estimator"])
        X = check_array(X)

        y_mean = self.predict(X)

        std_dev = np.std(self.residuals_)

        proba = np.zeros((X.shape[0], 2))

        # Iterate over predicted means to construct distribution information
        for i, mean in enumerate(y_mean):
            proba[i, :] = [mean, std_dev]

        return proba

    def predict(self, X):
        """Return predicted values for the given data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : array-like of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        check_is_fitted(self, ["_estimator"])
        X = check_array(X)

        return self._estimator.predict(X)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        param1 = {}
        param2 = {"fit_intercept": True, "normalize": False, "copy_X": True}
        return [param1, param2]
