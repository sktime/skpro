"""Johnson QPD wrapper to convert quantile predictions to probabilistic predictions."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["RakshithaKowlikar"]
__all__ = ["JohnsonQPDRegressor"]

import numpy as np
import pandas as pd
from sklearn import clone

from skpro.distributions import QPD_Johnson
from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class JohnsonQPDRegressor(BaseProbaRegressor):
    """Wrapper converting quantile predictions to probabilistic via Johnson QPD.

    The ``JohnsonQPDRegressor`` takes any quantile regressor and converts its
    quantile predictions into full probability distributions using the Johnson
    Quantile-Parameterized Distribution.

    The wrapper fits three copies of the base quantile regressor at different
    quantile levels (alpha, 0.5, 1-alpha), then uses these predictions to
    parameterize a Johnson QPD for each prediction.

    ``fit(X, y)`` - changes state by fitting three clones of ``estimator``
        with quantile parameters set to ``alpha``, 0.5, and ``1-alpha``.

    ``predict(X)`` - result is median predictions from the median quantile regressor.

    ``predict_proba(X)`` - first obtains quantile predictions from all three
        regressors, then returns a ``QPD_Johnson`` distribution parameterized
        by these quantiles.

    ``predict_interval(X)``, ``predict_quantiles(X)`` - as ``predict_proba``,
        using the base class default to obtain predictions from the
        ``QPD_Johnson`` distribution.

    Parameters
    ----------
    estimator : sklearn regressor
        quantile regressor to fit at different quantile levels,
        must have a ``quantile`` parameter that can be set via ``set_params``
    alpha : float, default=0.1
        quantile level for symmetric percentile triplet,
        must be in the range (0, 0.5),
        the wrapper will fit quantiles at ``alpha``, 0.5, and ``1-alpha``
    lower : float, optional, default=None
        lower bound for distribution support, if None, unbounded below
    upper : float, optional, default=None
        upper bound for distribution support, if None, unbounded above
    base_dist : str, default='normal'
        base distribution type for Johnson QPD,
        one of 'normal', 'logistic', 'cauchy'

    Attributes
    ----------
    estimator_low_ : sklearn regressor
        clone of ``estimator``, fitted to lower quantile (alpha)
    estimator_median_ : sklearn regressor
        clone of ``estimator``, fitted to median quantile (0.5)
    estimator_high_ : sklearn regressor
        clone of ``estimator``, fitted to upper quantile (1-alpha)

    Examples
    --------
    >>> from skpro.regression.compose import JohnsonQPDRegressor
    >>> from sklearn.linear_model import QuantileRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> # Create wrapper with quantile regressor
    >>> base_model = QuantileRegressor(alpha=0.0)
    >>> proba_model = JohnsonQPDRegressor(base_model, alpha=0.1)
    >>>
    >>> # Fit and predict
    >>> proba_model.fit(X_train, y_train)
    JohnsonQPDRegressor(...)
    >>> y_pred_proba = proba_model.predict_proba(X_test)
    >>> y_pred_interval = proba_model.predict_interval(
    ...     X_test, coverage=0.9
    ... )
    """

    _tags = {
        "capability:missing": False,
        "capability:survival": False,
        "capability:quantile": True,
    }

    def __init__(
        self, estimator, alpha=0.1, lower=None, upper=None, base_dist="normal"
    ):
        self.estimator = estimator
        self.alpha = alpha
        self.lower = lower
        self.upper = upper
        self.base_dist = base_dist
        super().__init__()

    def _fit(self, X, y, C=None):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pd.DataFrame, must be same length as X
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
        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        # validate alpha range
        if not 0 < self.alpha < 0.5:
            raise ValueError(
                f"alpha must be in the range (0, 0.5), but found alpha={self.alpha}"
            )

        # convert y to 1D array for sklearn compatibility
        if isinstance(y, pd.DataFrame):
            # check for multivariate targets
            if y.shape[1] > 1:
                raise ValueError(
                    "JohnsonQPDRegressor supports only univariate targets, "
                    f"but y has {y.shape[1]} columns."
                )
            y_array = y.values.ravel()
        elif isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y.ravel() if hasattr(y, "ravel") else y

        # clone estimator for three quantile levels
        self.estimator_low_ = clone(self.estimator)
        self.estimator_median_ = clone(self.estimator)
        self.estimator_high_ = clone(self.estimator)

        # set quantile parameter for each estimator
        self.estimator_low_.set_params(quantile=self.alpha)
        self.estimator_median_.set_params(quantile=0.5)
        self.estimator_high_.set_params(quantile=1 - self.alpha)

        # fit all three estimators
        self.estimator_low_.fit(X, y_array)
        self.estimator_median_.fit(X, y_array)
        self.estimator_high_.fit(X, y_array)

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted" = self.is_fitted=True

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in fit
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`, same columns as `y` in `fit`
            labels predicted for `X`, median (0.5 quantile) predictions
        """
        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        y_pred = self.estimator_median_.predict(X)

        # ensure return is a DataFrame with correct columns
        if isinstance(y_pred, pd.DataFrame):
            y_pred_df = y_pred.copy()
            y_pred_df.columns = self._y_metadata["feature_names"]
        else:
            y_pred_arr = np.asarray(y_pred)
            if y_pred_arr.ndim == 1:
                y_pred_arr = y_pred_arr.reshape(-1, 1)
            y_pred_df = pd.DataFrame(
                y_pred_arr, index=X.index, columns=self._y_metadata["feature_names"]
            )

        return y_pred_df

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in fit
            data to predict labels for

        Returns
        -------
        y_pred : skpro BaseDistribution, same length as X
            labels predicted for X
        """
        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        # Get quantile predictions
        qv_low = self.estimator_low_.predict(X)
        qv_median = self.estimator_median_.predict(X)
        qv_high = self.estimator_high_.predict(X)

        # convert predictions to arrays
        if isinstance(qv_low, pd.DataFrame):
            qv_low = qv_low.values.flatten()
        elif isinstance(qv_low, pd.Series):
            qv_low = qv_low.values

        if isinstance(qv_median, pd.DataFrame):
            qv_median = qv_median.values.flatten()
        elif isinstance(qv_median, pd.Series):
            qv_median = qv_median.values

        if isinstance(qv_high, pd.DataFrame):
            qv_high = qv_high.values.flatten()
        elif isinstance(qv_high, pd.Series):
            qv_high = qv_high.values

        # enforce quantile ordering and minimum spread
        # if quantile estimates are degenerate or misordered,
        # use symmetric spread around median with minimum width
        q_range = np.abs(qv_high - qv_low)
        min_spread = np.maximum(q_range * 0.01, 1e-3)
        spread = np.maximum(min_spread, q_range / 2.0)

        qv_low_fixed = qv_median - spread
        qv_high_fixed = qv_median + spread

        # apply bounds if specified
        if self.lower is not None:
            qv_low_fixed = np.maximum(qv_low_fixed, self.lower)
            qv_median = np.maximum(qv_median, self.lower)
            qv_high_fixed = np.maximum(qv_high_fixed, self.lower)

        if self.upper is not None:
            qv_low_fixed = np.minimum(qv_low_fixed, self.upper)
            qv_median = np.minimum(qv_median, self.upper)
            qv_high_fixed = np.minimum(qv_high_fixed, self.upper)

        # construct QPD_Johnson distribution
        dist = QPD_Johnson(
            alpha=self.alpha,
            qv_low=qv_low_fixed,
            qv_median=qv_median,
            qv_high=qv_high_fixed,
            lower=self.lower,
            upper=self.upper,
            base_dist=self.base_dist,
            index=X.index,
            columns=self._y_metadata["feature_names"],
        )

        return dist

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return "default" set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.linear_model import QuantileRegressor

        params1 = {
            "estimator": QuantileRegressor(alpha=0.0),
            "alpha": 0.1,
        }

        params2 = {
            "estimator": QuantileRegressor(alpha=0.0),
            "alpha": 0.25,
            "lower": 0.0,
        }

        return [params1, params2]
