# -*- coding: utf-8 -*-
"""Residual regression - one regressor for mean, one for scale."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
from sklearn import clone

from skpro.distributions.normal import Normal
from skpro.regression.base import BaseProbaRegressor


class ResidualDouble(BaseProbaRegressor):
    """Residual double regressor.

    One regressor predicting the mean, and one the deviation from the mean.

    TODO - math description

    Parameters
    ----------
    estimator : skpro estimator, BaseProbaRegressor descendant
        estimator predicting the mean or location
    estimator_resid : skpro estimator, BaseProbaRegressor descendant, optional
        estimator predicting the scale of the residual
        default = sklearn DummyRegressor(strategy="mean")

    TODO - add
    estimator_resid : skpro estimator or dict of estimators with str keys
    residual_trafo : str, or transformer, default="absolute"
        determines the labels predicted by ``estimator_resid``
        absolute = absolute residuals
        squared = squared residuals
    distr_type : str or BaseDistribution, default = "Normal"
        type of distribution to predict
        str options are "Normal", "Laplace", "Cauchy", "t"
    use_y_pred : bool, default=False
        whether to use the predicted location in predicting the scale of the residual
    cv : optional, sklearn cv splitter, default = None
        if passed, will be used to obtain out-of-sample residuals according to cv
        instead of in-sample residuals in ``fit`` of this estimator
    min_scale : float, default=1e-10
        minimum scale parameter if ``estimator_resid`` is an estimator (not dict)

    Example
    -------
    >>> from skpro.regression.residual import ResidualDouble
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> reg_mean = LinearRegression()
    >>> reg_resid = RandomForestRegressor()
    >>> reg_proba = ResidualDouble(reg_mean, reg_resid)
    >>>
    >>> reg_proba.fit(X, y)
    ResidualDouble(...)
    >>> y_pred_mean = reg_proba.predict(X)
    >>> y_pred_proba = reg_proba.predict_proba(X)
    """

    _tags = {"capability:missing": True}

    def __init__(self, estimator, estimator_resid=None, min_scale=1e-10):

        self.estimator = estimator
        self.estimator_resid = estimator_resid
        self.min_scale = min_scale

        super(ResidualDouble, self).__init__()

        self.estimator_ = clone(estimator)

        if estimator_resid is None:
            from sklearn.dummy import DummyRegressor

            self.estimator_resid_ = DummyRegressor(strategy="mean")
        else:
            self.estimator_resid_ = clone(estimator_resid)

    def _fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        est = self.estimator_
        est_r = self.estimator_resid_

        self._y_cols = y.columns
        y = y.values.flatten()

        est.fit(X, y)
        resids = np.abs(y - est.predict(X))

        resids = resids.flatten()

        est_r.fit(X, resids)

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted" = self.is_fitted=True

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
            labels predicted for `X`
        """
        est = self.estimator_

        y_pred = est.predict(X)
        y_pred = pd.DataFrame(y_pred, columns=self._y_cols, index=X.index)

        return y_pred

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        est = self.estimator_
        est_r = self.estimator_resid_
        min_scale = self.min_scale

        y_pred_loc = est.predict(X)
        y_pred_loc = y_pred_loc.reshape(-1, 1)

        y_pred_scale = est_r.predict(X)
        y_pred_scale = y_pred_scale.clip(min=min_scale)
        y_pred_scale = y_pred_scale.reshape(-1, 1)

        y_pred = Normal(
            mu=y_pred_loc, sigma=y_pred_scale, index=X.index, columns=self._y_cols
        )

        return y_pred

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
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression

        params1 = {"estimator": RandomForestRegressor()}
        params2 = {
            "estimator": LinearRegression(),
            "estimator_resid": RandomForestRegressor(),
            "min_scale": 1e-7,
        }

        return [params1, params2]
