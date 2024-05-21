"""Probabilistic regression by bootstrap."""

__author__ = ["fkiraly"]
__all__ = ["EnbpiRegressor"]

import numpy as np
import pandas as pd
from sklearn import clone
from sklearn.utils import check_random_state

from skpro.distributions.empirical import Empirical
from skpro.regression.base import BaseProbaRegressor
from skpro.utils.numpy import flatten_to_1D_if_colvector
from skpro.utils.sklearn import prep_skl_df


class EnbpiRegressor(BaseProbaRegressor):
    r"""EnbPI probabilistic regressor, for conformal prediction intervals.

    Wraps an ``sklearn`` regressor and turns it into an ``skpro`` regressor
    with access to all probabilistic prediction methods.

    Follows the original algorithm in [1]_, in ``predict_proba`` a distribution
    implied by the quantile predictions is returned, an ``Empirical`` distribution.

    Fits ``n_estimators`` clones of a tabular ``sklearn`` regressor on
    datasets which are bootstrap sub-samples, i.e.,
    independent row samples with replacement.

    The clones are aggregated to predict quantiles of the target distribution,
    following the original algorithm in [1]_. The parameters in the reference
    map as follows: :math:`\mathcal{A}` is ``estimator``, :math:`B` is
    ``n_bootstrap_samples``, :math:`x_i, i = 1, \dots, T` are the rows of
    ``X`` in ``fit``, :math:`y_i, i = 1, \dots, T` are the rows of ``y`` in ``fit``,
    :math:`x_i, i = T+1, \dots, T + T_1` are the rows of ``X`` in ``predict_interval``,
    :math:`\phi` is ``agg_fun``.

    The :math:`C_{T, t}^{\phi, \alpha}(x_t), t = T+1, \dots, T + T_1` are encoded
    in the prediction object returned by ``predict_interval`` or ``predict_proba``.

    Parameters
    ----------
    estimator : sklearn regressor
        regressor to use in the bootstrap
    n_bootstrap_samples : int, default=100
        The number of bootstrap samples drawn
        If int, then indicates number of instances precisely
        Note: this is not the same as the size of each bootstrap sample.
        The size of the bootstrap sample is always equal to X.
    agg_fun : str or callable, default="mean"
        function to aggregate the predictions of the bootstrap samples
        If str, must be one of "mean" or "median", meaning np.mean or np.median
        If callable, must be a function of signature (n, m) -> m, or (n) -> 1
    symmetrize : bool, default=True
        whether to symmetrize the prediction intervals and predictive quantiles
        Default = True leads to the original algorithm in [1]_.
        If False, the conformalized sample and the prediction intervals are not
        symmetrized.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number generator;
        If ``RandomState`` instance, ``random_state`` is the random number generator;
        If None, the random number generator is the ``RandomState`` instance used
        by ``np.random``.

    Attributes
    ----------
    estimators_ : list of of skpro regressors
        clones of regressor in `estimator` fitted in the ensemble

    References
    ----------
    .. [1] Xu, Chen and Yao Xie (2021).
      Conformal prediction interval for dynamic time-series.
      The Proceedings of the 38th International Conference on Machine Learning,
      PMLR 139, 2021.

    Examples
    --------
    >>> from skpro.regression.enbpi import EnbpiRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg_tabular = LinearRegression()
    >>>
    >>> reg_proba = EnbpiRegressor(reg_tabular)
    >>> reg_proba.fit(X_train, y_train)
    EnbpiRegressor(...)
    >>> y_pred = reg_proba.predict_proba(X_test)
    """

    _tags = {
        "authors": ["fkiraly", "hamrel-cxu"],
        "capability:missing": True,
    }

    def __init__(
        self,
        estimator,
        n_bootstrap_samples=100,
        agg_fun="mean",
        symmetrize=True,
        random_state=None,
    ):
        self.estimator = estimator
        self.n_bootstrap_samples = n_bootstrap_samples
        self.agg_fun = agg_fun
        self.symmetrize = symmetrize
        self.random_state = random_state
        self._random_state = check_random_state(random_state)

        super().__init__()

        # todo: find the equivalent tag in sklearn for missing data handling
        # tags_to_clone = ["capability:missing"]
        # self.clone_tags(estimator, tags_to_clone)

        aggfun_dict = {
            "mean": np.nanmean,
            "median": np.nanmedian,
        }
        if agg_fun is None:
            self._agg_fun = np.nanmean
        elif agg_fun in aggfun_dict:
            self._agg_fun = aggfun_dict[agg_fun]
        else:
            self._agg_fun = agg_fun

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
        estimator = self.estimator
        n_bootstrap_samples = self.n_bootstrap_samples
        np.random.seed(self.random_state)

        inst_ix = X.index
        n = len(inst_ix)

        self.estimators_ = []
        self._cols = y.columns

        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        y_pred_insample = np.ones((n_bootstrap_samples,) + y.shape) * np.nan

        for i in range(n_bootstrap_samples):
            esti = clone(estimator)
            row_iloc = pd.RangeIndex(n)
            row_ss = _random_ss_ix(
                row_iloc, size=n, replace=True, random_state=self._random_state
            )
            inst_ix_i = inst_ix[row_ss]

            Xi = X.loc[inst_ix_i]
            Xi = Xi.reset_index(drop=True)

            yi = y.loc[inst_ix_i].values
            yi = flatten_to_1D_if_colvector(yi)

            self.estimators_ += [esti.fit(Xi, yi)]

            y_pred = esti.predict(Xi)
            y_pred_insample[[i], row_ss] = _coerce_numpy2d(y_pred)

        spl = self._agg_preds(y_pred_insample)

        errs = y.values - spl
        if self.symmetrize:
            errs = np.abs(errs)

        errs.sort(axis=0)
        self._errs = errs
        return self

    def _agg_preds(self, preds):
        """Aggregate predictions of bootstrap samples.

        Handles NaNs in the aggregated predictions
        by replacing them with the aggregated value from the same column
        (index of target variable), after the aggregation.

        Parameters
        ----------
        preds : np.ndarray, shape (n_bootstrap_samples, n_samples, n_targets)
            predictions of bootstrap samples

        Returns
        -------
        agg_preds : np.ndarray, shape (n_samples, n_targets)
            aggregated predictions
        """
        spl = self._agg_fun(preds, axis=0)
        nans = np.isnan(spl)
        if nans.any():
            for j in range(spl.shape[1]):
                colj = spl[:, j]
                colj_agg = self._agg_fun(colj)
                colj[nans[:, j]] = colj_agg
                spl[:, j] = colj
        return spl

    def _predict_proba(self, X) -> np.ndarray:
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
        cols = self._cols  # number of targets

        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)
        n = len(X)

        y_preds = np.zeros((len(self.estimators_), n, len(cols)))

        for i, est in enumerate(self.estimators_):
            y_preds[i] = _coerce_numpy2d(est.predict(X))

        # y_preds now has shape (n_estimators, n_samples, len(cols))
        # or, if symmetrize, (2 * n_estimators, n_samples, len(cols))
        y_preds_agg = self._agg_preds(y_preds)
        # y_preds_agg now has shape (n_samples, len(cols))

        if self.symmetrize:
            errs = np.concatenate([self._errs, -self._errs], axis=0)
        else:
            errs = self._errs

        n_emp_spl = len(errs)
        y_preds_rep = np.tile(y_preds_agg, (n_emp_spl, 1))

        errs = np.repeat(errs, n, axis=0)
        emp_df_vals = y_preds_rep + errs

        emp_ix = pd.MultiIndex.from_product([range(n_emp_spl), X.index])

        spl_df = pd.DataFrame(emp_df_vals, index=emp_ix, columns=cols)
        y_proba = Empirical(spl_df)
        return y_proba

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
        from sklearn.linear_model import LinearRegression

        params1 = {"estimator": LinearRegression()}
        params2 = {
            "estimator": LinearRegression(),
            "n_bootstrap_samples": 10,
        }
        params3 = {
            "estimator": LinearRegression(),
            "n_bootstrap_samples": 12,
            "agg_fun": "median",
            "symmetrize": False,
        }

        return [params1, params2, params3]


def _random_ss_ix(ix, size, replace=True, random_state=None):
    """Randomly uniformly sample indices from a list of indices."""
    if random_state is None:
        random_state = np.random.RandomState()

    a = range(len(ix))
    ixs = ix[random_state.choice(a, size=size, replace=replace)]
    return ixs


def _coerce_numpy2d(x):
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x
