"""Stacking ensemble of heterogeneous probabilistic regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Ashish-Kumar-Dash"]
__all__ = ["StackingProbaRegressor"]

import numpy as np
import pandas as pd

from skpro.base import BaseMetaEstimator
from skpro.regression.base import BaseProbaRegressor


class StackingProbaRegressor(BaseMetaEstimator, BaseProbaRegressor):
    """Stacking ensemble of heterogeneous probabilistic regressors.

    Fits multiple base probabilistic regressors and a final meta-learner.
    During ``fit``, out-of-fold distributional predictions (mean and variance)
    from the base regressors are used as meta-features to train the
    final estimator. During ``predict_proba``, base regressor predictions are
    transformed into meta-features and passed to the final estimator.

    Generalizes ``sklearn``'s ``StackingRegressor`` to the probabilistic
    regression setting, where predictions are full distributions
    rather than point predictions.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples or list of estimators
        The base ensemble members. Each estimator must be a descendant of
        ``BaseProbaRegressor``.
        If a plain list of estimators is passed, names are generated
        automatically from class names.
    final_estimator : BaseProbaRegressor, optional, default=None
        The meta-learner trained on distributional meta-features.
        If ``None``, defaults to ``ResidualDouble(LinearRegression())``.
    cv : int or sklearn cv splitter, optional, default=5
        Cross-validation strategy for generating out-of-fold predictions.
        If int, uses ``KFold(n_splits=cv)``.
    passthrough : bool, optional, default=False
        If ``True``, the original features ``X`` are appended to the
        meta-features before training the final estimator.

    Attributes
    ----------
    estimators_ : list of (str, estimator) tuples
        Fitted clones of the base estimators, re-fitted on full training data.
    final_estimator_ : BaseProbaRegressor
        Fitted clone of the final estimator.

    Examples
    --------
    >>> from skpro.regression.ensemble import StackingProbaRegressor
    >>> from skpro.regression.residual import ResidualDouble
    >>> from skpro.regression.linear import BayesianRidge
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg1 = ResidualDouble(LinearRegression())
    >>> reg2 = BayesianRidge()
    >>>
    >>> stacker = StackingProbaRegressor(
    ...     estimators=[("r1", reg1), ("r2", reg2)],
    ...     final_estimator=ResidualDouble(LinearRegression()),
    ... )
    >>> stacker.fit(X_train, y_train)
    StackingProbaRegressor(...)
    >>> y_pred = stacker.predict_proba(X_test)
    """

    _tags = {
        "object_type": "regressor_proba",
        "estimator_type": "regressor_proba",
        "named_object_parameters": "_estimators",
        "fitted_named_object_parameters": "estimators_",
        "capability:missing": True,
        "capability:survival": True,
    }

    def __init__(self, estimators, final_estimator=None, cv=5, passthrough=False):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.passthrough = passthrough

        super().__init__()

        est_list = self._estimators

        self._anytagis_then_set("capability:missing", False, True, est_list)
        self._anytagis_then_set("capability:survival", True, False, est_list)

    @property
    def _estimators(self):
        """Return base estimators as named object tuples."""
        return self._coerce_to_named_object_tuples(self.estimators, clone=False)

    @_estimators.setter
    def _estimators(self, value):
        self.estimators = value

    def _fit(self, X, y, C=None):
        """Fit base regressors via cross-validation and train meta-learner.

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to
        C : pandas DataFrame, optional (default=None)
            censoring information for survival analysis

        Returns
        -------
        self : reference to self
        """
        from sklearn.model_selection import KFold

        cv = self.cv
        if isinstance(cv, int):
            cv = KFold(n_splits=cv)

        final_estimator = self.final_estimator
        if final_estimator is None:
            from sklearn.linear_model import LinearRegression

            from skpro.regression.residual import ResidualDouble

            final_estimator = ResidualDouble(LinearRegression())

        est_list = self._estimators

        oof_means = {}
        oof_vars = {}
        for name, _ in est_list:
            oof_means[name] = pd.DataFrame(np.nan, index=X.index, columns=y.columns)
            oof_vars[name] = pd.DataFrame(np.nan, index=X.index, columns=y.columns)

        for tr_idx, tt_idx in cv.split(X):
            X_train = X.iloc[tr_idx]
            X_test = X.iloc[tt_idx]
            y_train = y.iloc[tr_idx]
            C_train = C.iloc[tr_idx] if C is not None else None

            for name, est in est_list:
                fitted = est.clone().fit(X_train, y_train, C=C_train)
                dist = fitted.predict_proba(X_test)
                test_idx = X_test.index
                oof_means[name].loc[test_idx] = dist.mean().values
                oof_vars[name].loc[test_idx] = dist.var().values

        X_meta = self._build_meta_features(oof_means, oof_vars)

        if self.passthrough:
            X_meta = pd.concat([X, X_meta], axis=1)
            X_meta.columns = X_meta.columns.astype(str)

        self.final_estimator_ = final_estimator.clone()
        self.final_estimator_.fit(X_meta, y, C=C)

        self.estimators_ = []
        for name, est in est_list:
            fitted_est = est.clone().fit(X, y, C=C)
            self.estimators_.append((name, fitted_est))

        return self

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in ``fit``
            data to predict labels for

        Returns
        -------
        y_pred : skpro BaseDistribution, same length as ``X``
            probabilistic predictions from the stacking meta-learner
        """
        means = {}
        vars_ = {}
        for name, est in self.estimators_:
            dist = est.predict_proba(X)
            means[name] = dist.mean()
            vars_[name] = dist.var()

        X_meta = self._build_meta_features(means, vars_)

        if self.passthrough:
            X_meta = pd.concat([X, X_meta], axis=1)
            X_meta.columns = X_meta.columns.astype(str)

        return self.final_estimator_.predict_proba(X_meta)

    def _build_meta_features(self, means, vars_):
        """Build meta-feature DataFrame from distributional statistics.

        Parameters
        ----------
        means : dict of {str: pd.DataFrame}
            Mean predictions keyed by estimator name.
        vars_ : dict of {str: pd.DataFrame}
            Variance predictions keyed by estimator name.

        Returns
        -------
        X_meta : pd.DataFrame
            Meta-feature matrix with columns named by estimator and statistic.
        """
        meta_dfs = []
        for name in means:
            m = means[name].copy()
            v = vars_[name].copy()
            m.columns = [f"{name}_mean_{c}" for c in m.columns]
            v.columns = [f"{name}_var_{c}" for c in v.columns]
            meta_dfs.append(m)
            meta_dfs.append(v)
        return pd.concat(meta_dfs, axis=1)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sklearn.linear_model import LinearRegression

        from skpro.regression.linear import BayesianRidge
        from skpro.regression.residual import ResidualDouble

        reg1 = ResidualDouble(LinearRegression())
        reg2 = BayesianRidge()

        params1 = {
            "estimators": [("r1", reg1), ("r2", reg2)],
            "cv": 3,
        }
        params2 = {
            "estimators": [("r1", reg1), ("r2", reg2)],
            "final_estimator": BayesianRidge(),
            "cv": 3,
        }
        params3 = {
            "estimators": [reg1, reg2],
            "passthrough": True,
            "cv": 3,
        }

        return [params1, params2, params3]
