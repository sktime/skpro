"""Residual regression - one regressor for mean, one for scale."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd
from sklearn import clone

from skpro.regression.base import BaseProbaRegressor
from skpro.utils.numpy import flatten_to_1D_if_colvector
from skpro.utils.sklearn import prep_skl_df


class ResidualDouble(BaseProbaRegressor):
    """Residual double regressor.

    Make a parametric probabilistic prediction using two tabular regressors, with
    one tabular regressor predicting the mean, and one the deviation from the mean.

    The mean is predicted by ``estimator``. The residual is predicted by
    ``estimator_resid``. The residual is transformed by ``residual_trafo``.
    The predicted mean and residual are passed to a distribution specified by
    ``distr_type``, and possibly ``distr_params``, ``distr_loc_scale_name``.

    The residuals predicted on the training data are used to fit
    ``estimator_resid``. If ``cv`` is passed, the residuals are out-of-sample
    according to ``cv``, otherwise in-sample.

    ``use_y_pred`` determines whether the predicted mean is used as a feature
    in predicting the residual.

    A formal description of the algorithm follows.

    In ``fit``, given training data ``X``, ``y``:

    1. Fit clone ``estimator_`` of ``estimator`` to predict ``y`` from ``X``,
       i.e., ``fit(X, y)``.
    2. Predict mean label ``y_pred`` for ``X`` using a clone of ``estimator``.
       If ``cv`` is ``None``, this is via plain ``estimator.predict(X)``.
       If ``cv`` is not ``None``, out-of-sample predictions are obtained via ``cv``.
       In this case, indices not appearing in ``cv`` are predicted in-sample.
    3. Compute residual ``resid`` as ``residual_trafo(y - y_pred)``.
       If ``residual_trafo`` is a transformer, ``residual_trafo.fit_transform`` is used.
    4. Fit clone ``estimator_resid_`` of ``estimator_resid``
       to predict ``resid`` from ``X``, i.e., ``fit(X, resid)``.
       If ``use_y_pred`` is ``True``, ``y_pred`` is used as a feature in predicting.

    In ``predict``, given test data ``X``:

    1. Predict mean label ``y_pred`` for ``X`` using ``estimator_.predict(X)``.
    2. Return ``y_pred``.

    In ``predict_proba``, given test data ``X``:

    1. Predict mean label ``y_pred`` for ``X`` using ``estimator_.predict(X)``.
    2. Predict residual ``resid`` for ``X`` using ``estimator_resid_.predict(X)``.
       If ``use_y_pred`` is ``True``, ``y_pred`` is used as a feature in predicting.
    3. Predict distribution ``y_pred_proba`` for ``X`` as follows:
       The location parameter is ``y_pred``. The scale parameter is ``resid``.
       Further parameters can be specified via ``distr_params``.
    4. Return ``y_pred_proba``.

    Parameters
    ----------
    estimator : sklearn regressor
        estimator predicting the mean or location
    estimator_resid : sklearn regressor
        estimator predicting the scale of the residual
        default = sklearn DummyRegressor(strategy="mean")
    residual_trafo : str, or transformer, default="absolute"
        determines the labels predicted by ``estimator_resid``
        absolute = absolute residuals
        squared = squared residuals
        if transformer, applies fit_transform to batch of signed residuals
    distr_type : str or BaseDistribution, default = "Normal"
        type of distribution to predict
        str options are "Normal", "Laplace", "Cauchy", "t"
    distr_loc_scale_name : tuple of length two, default = ("loc", "scale")
        names of the parameters in the distribution to use for location and scale
        if ``distr_type`` is a string, this is overridden to the correct parameters
        if ``distr_type`` is a BaseDistribution, this is used to determine the
        location and scale parameters that the predictions are passed to
    distr_params : dict, default = {}
        parameters to pass to the distribution
        must be valid parameters of ``distr_type``, if ``BaseDistribution``
        must be default or dict with key ``df``, if ``t`` distribution
    use_y_pred : bool, default=False
        whether to use the predicted location in predicting the scale of the residual
    cv : optional, sklearn cv splitter, default = None
        if passed, will be used to obtain out-of-sample residuals according to cv
        instead of in-sample residuals in ``fit`` of this estimator
    min_scale : float, default=1e-10
        minimum scale parameter. If smaller scale parameter is predicted by
        ``estimator_resid``, will be clipped to this value

    Attributes
    ----------
    estimator_ : sklearn regressor, clone of ``estimator``
        fitted estimator predicting the mean or location
    estimator_resid_ : sklearn regressor, clone of ``estimator_resid``
        fitted estimator predicting the scale of the residual

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

    def __init__(
        self,
        estimator,
        estimator_resid=None,
        residual_trafo="absolute",
        distr_type="Normal",
        distr_loc_scale_name=None,
        distr_params=None,
        use_y_pred=False,
        cv=None,
        min_scale=1e-10,
    ):
        self.estimator = estimator
        self.estimator_resid = estimator_resid
        self.residual_trafo = residual_trafo
        self.distr_type = distr_type
        self.distr_loc_scale_name = distr_loc_scale_name
        self.distr_params = distr_params
        self.use_y_pred = use_y_pred
        self.cv = cv
        self.min_scale = min_scale

        super().__init__()

        self.estimator_ = clone(estimator)

        if estimator_resid is None:
            from sklearn.dummy import DummyRegressor

            self.estimator_resid_ = DummyRegressor(strategy="mean")
        else:
            self.estimator_resid_ = clone(estimator_resid)

    def _predict_residuals_cv(self, X, y, cv, est):
        """Predict out-of-sample residuals for y from X using cv.

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to
        cv : sklearn cv splitter
            cv splitter to use for out-of-sample predictions

        Returns
        -------
        y_pred : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
            labels predicted for `X`
        """
        est = self.estimator_resid_
        method = "predict"
        y_pred = y.copy()

        for tr_idx, tt_idx in cv.split(X):
            X_train = X.iloc[tr_idx]
            X_test = X.iloc[tt_idx]
            y_train = y[tr_idx]
            fitted_est = clone(est).fit(X_train, y_train)
            y_pred[tt_idx] = getattr(fitted_est, method)(X_test)

        return y_pred

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
        residual_trafo = self.residual_trafo
        cv = self.cv
        use_y_pred = self.use_y_pred

        self._y_cols = y.columns

        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        # flatten column vector to 1D array to avoid sklearn complaints
        y = y.values
        y = flatten_to_1D_if_colvector(y)

        est.fit(X, y)

        if cv is None:
            y_pred = est.predict(X)
        else:
            y_pred = self._predict_residuals_cv(X, y, cv, est)

        if residual_trafo == "absolute":
            resids = np.abs(y - y_pred)
        elif residual_trafo == "squared":
            resids = (y - y_pred) ** 2
        else:
            resids = residual_trafo.fit_transform(y - y_pred)

        resids = flatten_to_1D_if_colvector(resids)

        if use_y_pred:
            y_ix = {"index": X.index, "columns": self._y_cols}
            X_r = pd.concat([X, pd.DataFrame(y_pred, **y_ix)], axis=1)
        else:
            X_r = X

        # coerce X to pandas DataFrame with string column names
        X_r = prep_skl_df(X_r, copy_df=True)

        est_r.fit(X_r, resids)

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

        X = prep_skl_df(X, copy_df=True)

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
        use_y_pred = self.use_y_pred
        distr_type = self.distr_type
        distr_loc_scale_name = self.distr_loc_scale_name
        distr_params = self.distr_params
        min_scale = self.min_scale

        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        n_cols = len(self._y_cols)

        if distr_params is None:
            distr_params = {}

        # predict location - this is the same as in _predict
        y_pred_loc = est.predict(X)
        y_pred_loc = y_pred_loc.reshape(-1, n_cols)

        # predict scale
        # if use_y_pred, use predicted location as feature
        if use_y_pred:
            y_ix = {"index": X.index, "columns": self._y_cols}
            X_r = pd.concat([X, pd.DataFrame(y_pred_loc, **y_ix)], axis=1)
        # if not use_y_pred, use only original features
        else:
            X_r = X

        # coerce X to pandas DataFrame with string column names
        X_r = prep_skl_df(X_r, copy_df=True)

        y_pred_scale = est_r.predict(X_r)
        y_pred_scale = y_pred_scale.clip(min=min_scale)
        y_pred_scale = y_pred_scale.reshape(-1, n_cols)

        # create distribution with predicted scale and location
        # we deal with string distr_types by getting class and param names
        if distr_type == "Normal":
            from skpro.distributions.normal import Normal

            distr_type = Normal
            distr_loc_scale_name = ("mu", "sigma")
        elif distr_type == "Laplace":
            from skpro.distributions.laplace import Laplace

            distr_type = Laplace
            distr_loc_scale_name = ("mu", "scale")
        elif distr_type in ["Cauchy", "t"]:
            from skpro.distributions.t import TDistribution

            distr_type = TDistribution
            distr_loc_scale_name = ("mu", "sigma")

        # collate all parameters for the distribution constructor
        # distribution params, if passed
        params = distr_params
        # row/column index
        ix = {"index": X.index, "columns": self._y_cols}
        params.update(ix)
        # location and scale
        loc_scale = {
            distr_loc_scale_name[0]: y_pred_loc,
            distr_loc_scale_name[1]: y_pred_scale,
        }
        params.update(loc_scale)

        # create distribution and return
        y_pred = distr_type(**params)
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
        from sklearn.model_selection import KFold

        params1 = {"estimator": RandomForestRegressor()}
        params2 = {
            "estimator": LinearRegression(),
            "estimator_resid": RandomForestRegressor(),
            "min_scale": 1e-7,
            "residual_trafo": "squared",
            "use_y_pred": True,
            "distr_type": "Laplace",
        }
        params3 = {
            "estimator": LinearRegression(),
            "estimator_resid": RandomForestRegressor(),
            "min_scale": 1e-6,
            "use_y_pred": True,
            "distr_type": "t",
            "distr_params": {"df": 3},
            "cv": KFold(n_splits=3),
        }
        params4 = {"estimator": RandomForestRegressor(), "cv": KFold(n_splits=3)}

        return [params1, params2, params3, params4]
