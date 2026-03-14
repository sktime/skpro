"""Meta-strategy for online learning: batch-mixture of fitted regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["patelchaitany"]
__all__ = ["OnlineBatchMixture"]

import pandas as pd

from skpro.distributions.mixture import Mixture
from skpro.regression.base import BaseProbaRegressor


class OnlineBatchMixture(BaseProbaRegressor):
    """Online regression by fitting separate regressors per batch and mixing.

    In ``fit``, fits a clone of the wrapped regressor on the initial data.
    In ``update``, fits a fresh clone on each new batch of data.
    In ``predict_proba``, returns a ``Mixture`` of the predictions from all
    fitted regressors, with weights proportional to the number of samples
    in each batch.

    Batches with fewer than ``min_batch_size`` samples are handled according
    to ``batch_mode``:

    * ``"accumulate"`` (default): small batches are buffered and pooled
      until the accumulated size reaches ``min_batch_size``, then a new
      regressor is fitted on the pooled data.
    * ``"discard"``: small batches are silently dropped.

    Caveat: data indices are reset to ``RangeIndex`` internally, even if
    some indices passed in ``fit`` and ``update`` overlap.

    Parameters
    ----------
    estimator : skpro regressor, descendant of ``BaseProbaRegressor``
        Blueprint regressor to be cloned and fitted on each batch.
    min_batch_size : int, default=1
        Minimum number of samples required to fit a new regressor on a batch.
        Batches smaller than this are handled according to ``batch_mode``.
    batch_mode : str, one of ``"accumulate"`` or ``"discard"``, default="accumulate"
        How to handle batches with fewer than ``min_batch_size`` samples.
        ``"accumulate"``: buffer small batches until ``min_batch_size`` is reached.
        ``"discard"``: drop small batches entirely.

    Attributes
    ----------
    estimators_ : list of skpro regressors
        Fitted regressor clones, one per processed batch.
    n_samples_ : list of int
        Number of samples in each processed batch. Used as ``Mixture`` weights.

    Examples
    --------
    >>> from skpro.regression.online import OnlineBatchMixture
    >>> from skpro.regression.residual import ResidualDouble
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg_proba = ResidualDouble(LinearRegression())
    >>> online = OnlineBatchMixture(reg_proba, min_batch_size=30)
    >>> online.fit(X_train[:200], y_train[:200])
    OnlineBatchMixture(...)
    >>> online.update(X_train[200:], y_train[200:])
    OnlineBatchMixture(...)
    >>> y_pred = online.predict_proba(X_test)
    """

    _tags = {
        "capability:update": True,
        "authors": ["patelchaitany"],
    }

    def __init__(self, estimator, min_batch_size=1, batch_mode="accumulate"):
        self.estimator = estimator
        self.min_batch_size = min_batch_size
        self.batch_mode = batch_mode

        super().__init__()

        tags_to_clone = [
            "capability:missing",
            "capability:survival",
        ]
        self.clone_tags(estimator, tags_to_clone)

    def _fit(self, X, y, C=None):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to
        C : pd.DataFrame, optional (default=None)
            censoring information for survival analysis,
            should have same column name as y, same length as X and y
            should have entries 0 and 1 (float or int)
            0 = uncensored, 1 = (right) censored
            if None, all observations are assumed to be uncensored
            Can be passed to any probabilistic regressor,
            but is ignored if capability:survival tag is False.

        Returns
        -------
        self : reference to self
        """
        estimator = self.estimator.clone()
        estimator.fit(X=X, y=y, C=C)

        self.estimators_ = [estimator]
        self.n_samples_ = [len(X)]

        self._X_buffer = None
        self._y_buffer = None
        self._C_buffer = None

        return self

    def _update(self, X, y, C=None):
        """Update regressor with new batch of training data.

        Fits a new regressor clone on the batch if it is large enough,
        or buffers/discards the data according to ``batch_mode``.

        State required:
            Requires state to be "fitted".

        Writes to self:
            Updates fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
            labels to fit regressor to
        C : pd.DataFrame, optional (default=None)
            censoring information for survival analysis,
            should have same column name as y, same length as X and y
            should have entries 0 and 1 (float or int)
            0 = uncensored, 1 = (right) censored
            if None, all observations are assumed to be uncensored
            Can be passed to any probabilistic regressor,
            but is ignored if capability:survival tag is False.

        Returns
        -------
        self : reference to self
        """
        X_pool = self._pool_data(self._X_buffer, X)
        y_pool = self._pool_data(self._y_buffer, y)
        C_pool = self._pool_data(self._C_buffer, C)

        n_pool = len(X_pool)

        if n_pool >= self.min_batch_size:
            estimator = self.estimator.clone()
            estimator.fit(X=X_pool, y=y_pool, C=C_pool)
            self.estimators_.append(estimator)
            self.n_samples_.append(n_pool)
            self._X_buffer = None
            self._y_buffer = None
            self._C_buffer = None
        elif self.batch_mode == "accumulate":
            self._X_buffer = X_pool
            self._y_buffer = y_pool
            self._C_buffer = C_pool
        else:
            self._X_buffer = None
            self._y_buffer = None
            self._C_buffer = None

        return self

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        Returns a ``Mixture`` of predictions from all fitted batch regressors,
        weighted by the number of samples each was trained on.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in ``fit``
            data to predict labels for

        Returns
        -------
        y_pred : skpro ``Mixture`` distribution, same length as ``X``
            mixture of probabilistic predictions from all batch regressors
        """
        y_probas = [est.predict_proba(X) for est in self.estimators_]
        weights = self.n_samples_

        return Mixture(y_probas, weights=weights)

    def _pool_data(self, X, X_new):
        """Pool existing buffer data with new batch data.

        Treats X_new as data with new indices, even if some indices overlap
        with X.

        Parameters
        ----------
        X : pandas DataFrame or None
            existing buffered data
        X_new : pandas DataFrame or None
            new batch data

        Returns
        -------
        X_pooled : pandas DataFrame or None
            concatenated data, with reset index
        """
        if X is None and X_new is None:
            return None
        if X is None and X_new is not None:
            return X_new.reset_index(drop=True)
        if X is not None and X_new is None:
            return X.reset_index(drop=True)
        X_pooled = pd.concat([X, X_new], ignore_index=True)
        return X_pooled

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
        from skbase.utils.dependencies import _check_estimator_deps
        from sklearn.linear_model import LinearRegression, Ridge

        from skpro.regression.residual import ResidualDouble
        from skpro.survival.coxph import CoxPH

        regressor = ResidualDouble(LinearRegression())

        params = [
            {"estimator": regressor},
            {
                "estimator": regressor,
                "min_batch_size": 5,
                "batch_mode": "discard",
            },
            {
                "estimator": regressor,
                "min_batch_size": 3,
                "batch_mode": "accumulate",
            },
        ]

        if _check_estimator_deps(CoxPH, severity="none"):
            params.append({"estimator": CoxPH()})
        else:
            params.append({"estimator": ResidualDouble(Ridge())})

        return params
