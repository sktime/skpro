"""Multiple quantile regression."""

__all__ = ["MultipleQuantileRegressor"]
__author__ = ["Ram0nB"]

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone

from skpro.regression.base import BaseProbaRegressor


class MultipleQuantileRegressor(BaseProbaRegressor):
    r"""Multiple quantile regressor.

    A tabular quantile regressor typically regresses a single probability level,
    however, often more than one quantile probability is of interest. Therefore, this
    multiple quantile regressor wraps multiple quantile regressors for multiple quantile
    probabilities so that each probability has one corresponding regressor. After
    fitting, all quantile regressors can be used to make probabilistic predictions.

    In `fit`, for every probability in alpha, the quantile_regressor is cloned and the
    probability is set. Subsequently all regressors are fitted.

    In probabilistic predict-like methods, if predictions of a quantile are requested,
    the fitted quantile regressor that is nearest to the desired quantile probability
    is used for the prediction, of the requested quantile.

    For instance, let :math:`\alpha = [\alpha_1, \alpha_2, \ldots, \alpha_n]`
    be the `alpha` provided to `__init__`, and
    let :math:`\alpha' = [\alpha'_1, \alpha'_2, \ldots, \alpha'_m]` be the quantiles
    requested in `predict_quantiles`.
    Then, we use the fitted quantile regressor at quantile :math:`\hat{\alpha}_j`,
    :math:`\hat{\alpha}_j := \underset{i = 1 \dots n}{\mathrm{argmin}}\ | \alpha'_j
    - \alpha_i |` to make the quantile prediction for the requested quantile
    probability :math:`\alpha'_j`.

    Consistently, the `predict_proba` method returns an empirical
    distribution with supports at quantile points corresponding to `alpha`,
    and weights corresponding to the nearest quantile regressor.

    Parameters
    ----------
    quantile_regressor : Sklearn compatible quantile regressor
        Tabular quantile regressor for probabilistic prediction methods.
    alpha_name : str
        Alpha parameter name that sets the quantile probability level for the
        quantile_regressor.
    alpha : list with float
        A list of probabilities in the open interval (0, 1).
        For each probability, a quantile_regressor will be fit.
    mean_regressor : Sklearn compatible regressor or None, default=None
        Tabular mean regressor for `predict`. Can't predict mean if not provided.
    n_jobs : int or None, default=None
        The number of jobs to run in parallel for `fit` and all probabilistic prediction
        methods. -1 means using all processors, -2 means using all except one processors
        and None means no parallelization.
    sort_quantiles : bool, default=False
        For the quantile estimates, the lack of monotinicity can be a problem caused by
        the crossing of quantiles, also known as the quantile crossing problem. This can
        happen because each quantile is predicted by a single independent instance of
        the quantile_regressor. To resolve this issue, the quantiles for all predictions
        per row index will be sorted if sort_quantiles=True. This methodology will never
        lead to a higher quantile loss and solves the quantile crossing problem [1]_.

    Attributes
    ----------
    regressors_ : dict
        All fitted regressors.
        The keys are the probabilities in alpha and "mean" for the mean.

    References
    ----------
    .. [1] Victor Chernozhukov, Iván Fernández-Val, and Alfred Galichon. Quantile and
       probability curves without crossing. Econometrica, 78(3):1093-1125, 2010.

    Examples
    --------
    >>> from sklearn.linear_model import QuantileRegressor, LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.datasets import load_diabetes
    >>> from skpro.regression.multiquantile import MultipleQuantileRegressor
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> alpha = [0.1, 0.3, 0.5, 0.7, 0.9]
    >>> mqr = MultipleQuantileRegressor(
    ...     mean_regressor=LinearRegression(),
    ...     quantile_regressor=QuantileRegressor(solver="highs"),
    ...     alpha_name="quantile",
    ...     alpha=alpha,
    ...     n_jobs=-1,
    ... )
    >>> mqr = mqr.fit(X_train, y_train)
    >>> predicted_quantiles = mqr.predict_quantiles(X_test, alpha)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ram0nb"],
        # estimator tags
        # --------------
        "capability:missing": False,
        "capability:multioutput": False,
    }

    def __init__(
        self,
        quantile_regressor,
        alpha_name,
        alpha,
        mean_regressor=None,
        n_jobs=None,
        sort_quantiles=False,
    ):
        self.quantile_regressor = quantile_regressor
        self.alpha_name = alpha_name
        self.alpha = alpha
        self.mean_regressor = mean_regressor
        self.n_jobs = n_jobs
        self.sort_quantiles = sort_quantiles

        # input checks
        self._no_mean_reg = True if mean_regressor is None else False
        if len(alpha) == 0:
            raise ValueError("at least one value in alpha is required.")
        elif np.amin(alpha) <= 0 or np.amax(alpha) >= 1:
            raise ValueError(
                "values in alpha must lie in the open interval (0, 1), "
                f"but found alpha: {alpha}."
            )

        super().__init__()

    def _fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, Series, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        # clone, set alpha and list all regressors
        regressors = [clone(self.mean_regressor)] if not self._no_mean_reg else []
        for a in self.alpha:
            q_est = clone(self.quantile_regressor)
            q_est = q_est.set_params(**{self.alpha_name: a})
            q_est_p = q_est.get_params()
            if q_est_p[self.alpha_name] != a:
                raise ValueError("Can't set alpha for quantile regressor.")
            else:
                regressors.append(q_est)

        # fit regressors
        def _fit_regressor(X, y, regressor):
            return regressor.fit(X=X, y=y)

        X_fit = X.values
        y_fit = y.values.flatten()
        fitted_regressors = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_regressor)(X_fit, y_fit, regressor) for regressor in regressors
        )

        # put fitted regressors in dict and write to self
        regressors_ = {}
        if not self._no_mean_reg:
            regressors_["mean"] = fitted_regressors.pop(0)
        for i, a in enumerate(self.alpha):
            regressors_[a] = fitted_regressors[i]
        self.regressors_ = regressors_

        # write varname to self ("capability:multioutput": False -> 1 column)
        self._y_varname = y.columns[0]

        return self

    def _predict(self, X):
        """Predict labels for data from features.

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
        y : pandas DataFrame, same length as `X`
            labels predicted for `X`
        """
        if self._no_mean_reg:
            raise ValueError("can't predict: no mean_regressor provided.")

        # predict
        X_pred = X.values
        preds = self.regressors_["mean"].predict(X_pred)

        # format predictions as DataFrame
        preds = np.array(preds).flatten()
        preds = pd.DataFrame(preds, columns=[self._y_varname])
        preds.index = X.index

        return preds

    def _predict_quantiles(self, X, alpha):
        """Compute/return quantile predictions.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and default _predict_interval

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for
        alpha : guaranteed list of float
            A list of probabilities at which quantile predictions are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from ``y`` in fit,
                second level being the values of alpha passed to the function.
            Row index is equal to row index of ``X``.
            Entries are quantile predictions, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        # per a in alpha, list fitted regressor that is nearest to a
        regressors = []
        for a in sorted(alpha):
            nearest_fitted_a = min(self.alpha, key=lambda p: abs(p - a))
            regressors.append(self.regressors_[nearest_fitted_a])

        # predict
        def _predict_quantile(X, regressor):
            pred = regressor.predict(X=X)
            pred = np.array(pred).flatten()
            return pred

        X_pred = X.values
        preds = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_quantile)(X_pred, regressor) for regressor in regressors
        )

        # format predictions as DataFrame
        y_cols = [self._y_varname]
        columns_sorted = pd.MultiIndex.from_product([y_cols, sorted(alpha)])
        columns = pd.MultiIndex.from_product([y_cols, alpha])
        index = X.index
        preds = pd.DataFrame(np.transpose(preds), columns=columns_sorted, index=index)

        # solve quantile crossing problem
        if self.sort_quantiles:
            preds = preds.apply(
                lambda row: row.sort_values().values, axis=1, result_type="broadcast"
            )

        # sort columns in the order of alpha requested
        preds = preds.reindex(columns=columns)

        return preds

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
        from skpro.distributions import Empirical

        alpha_sorted = sorted(self.alpha)

        # get quantile prediction for all fitted quantile regressors
        quantile_preds = self._predict_quantiles(X, alpha_sorted)

        # format as emprical sample for empirical distr
        # row multiindex: (alpha, X.index)
        # column index  : as y in fit
        empirical_spl = quantile_preds.stack(level=1).swaplevel(0, 1)

        # obtain alpha weights for empirical distr such that we take the nearest
        # available quantile prob (the cum weights match the chosen quantile prob)
        alpha_np = np.array(alpha_sorted)
        alpha_diff = np.diff(alpha_np)
        alpha_diff2 = np.repeat(alpha_diff, 2) / 2
        weight_double = np.concatenate([[alpha_np[0]], alpha_diff2, [1 - alpha_np[-1]]])
        weight_double2 = weight_double.reshape(-1, 2)
        weights = weight_double2.sum(axis=1)

        # obtain weights per emprical sample
        empirical_spl_weights = pd.Series(index=empirical_spl.index)
        for i, a in enumerate(alpha_sorted):
            empirical_spl_weights.loc[a] = weights[i]

        y_pred_proba = Empirical(
            empirical_spl,
            weights=empirical_spl_weights,
            index=X.index,
            columns=[self._y_varname],
        )

        return y_pred_proba

    @classmethod
    def get_test_params(cls):
        """
        Return testing parameter settings for the regressor.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression, QuantileRegressor

        from skpro.regression.tests.test_all_regressors import TEST_ALPHAS

        # extract all alpha values from TEST_ALPHAS
        alpha = []
        for a in TEST_ALPHAS:
            if isinstance(a, list):
                for a2 in a:
                    alpha.append(a2)
            else:
                alpha.append(a)

        # TEST_ALPHAS also used for predict_interval coverage, so we also need the
        # corresponding coverage quantile probabilities
        coverage_alpha = []
        for a in alpha:
            coverage_alpha.extend([0.5 - 0.5 * float(a), 0.5 + 0.5 * float(a)])
        alpha += coverage_alpha
        alpha = sorted(list(set(alpha)))

        params = [
            {
                "mean_regressor": LinearRegression(),
                "quantile_regressor": QuantileRegressor(solver="highs"),
                "alpha_name": "quantile",
                "alpha": alpha,
                "n_jobs": -1,
                "sort_quantiles": True,
            },
            {
                "mean_regressor": GradientBoostingRegressor(),
                "quantile_regressor": GradientBoostingRegressor(loss="quantile"),
                "alpha_name": "alpha",
                "alpha": alpha,
                "n_jobs": -1,
                "sort_quantiles": True,
            },
        ]

        return params
