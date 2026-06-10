"""Adapter for River online regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["patelchaitany"]
__all__ = ["RiverRegressor"]

from copy import deepcopy

import pandas as pd

from skpro.regression.adapters.river._clone import _RiverDeepcopyCloner
from skpro.regression.adapters.river._utils import (
    _learn_batch,
    _predict_batch,
    is_river_estimator,
)
from skpro.regression.base import BaseOnlineRegressor


class RiverRegressor(BaseOnlineRegressor):
    """Adapter for River online regressors to the skpro point-prediction API.

    Wraps a River regressor and exposes ``fit``, ``update``, and ``predict``
    with skpro datatype conversion on the public interface. Probabilistic
    prediction is intentionally not provided; use a meta-estimator such as
    ``BaggingRegressor`` for distributional output.

    On ``fit`` and ``update``, the River model is trained incrementally via
    ``learn_many`` when available, otherwise ``learn_one``.

    Parameters
    ----------
    estimator : river.base.Estimator
        River regressor instance to wrap.

    Attributes
    ----------
    estimator_ : river estimator
        Deep copy of the River model after fitting.

    Examples
    --------
    >>> from skpro.regression.adapters.river import RiverRegressor
    >>> from river import linear_model
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> y_train = y_train.to_frame("target")
    >>>
    >>> reg = RiverRegressor(linear_model.LinearRegression())
    >>> reg.fit(X_train, y_train)
    RiverRegressor(...)
    >>> reg.update(X_test[:10], y_test[:10].to_frame("target"))
    RiverRegressor(...)
    >>> y_pred = reg.predict(X_test[10:])
    """

    _tags = {
        "authors": ["patelchaitany"],
        "maintainers": ["patelchaitany", "fkiraly"],
        "python_dependencies": "river",
        "capability:update": True,
        "capability:pred_int": False,
        "capability:multioutput": False,
        "capability:missing": False,
        "tests:vm": True,
    }

    def __init__(self, estimator):
        if not is_river_estimator(estimator):
            raise TypeError(
                "estimator must be a River estimator instance, "
                f"but found type {type(estimator)}"
            )
        self.estimator = estimator
        super().__init__()

    @classmethod
    def _get_clone_plugins(cls):
        parent_plugins = super()._get_clone_plugins()
        if parent_plugins is None:
            parent_plugins = []
        return [_RiverDeepcopyCloner] + list(parent_plugins)

    def _fit(self, X, y, C=None):
        if len(y.columns) != 1:
            raise ValueError(
                "RiverRegressor supports single-output regression only, "
                f"but y has {len(y.columns)} columns."
            )

        self.estimator_ = deepcopy(self.estimator)
        _learn_batch(self.estimator_, X, y)
        return self

    def _update(self, X, y, C=None):
        if len(y.columns) != 1:
            raise ValueError(
                "RiverRegressor supports single-output regression only, "
                f"but y has {len(y.columns)} columns."
            )

        _learn_batch(self.estimator_, X, y)
        return self

    def _predict(self, X):
        preds = _predict_batch(self.estimator_, X)
        columns = self._get_columns(method="predict")
        return pd.DataFrame(preds, index=X.index, columns=columns)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from river import linear_model

        params1 = {"estimator": linear_model.LinearRegression()}
        return [params1]
