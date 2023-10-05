"""Multiple quantile regression"""

__all__ = ["MultipleQuantileRegressor"]
__author__ = ["Ram0nB"]

import pandas as pd
import numpy as np

from skpro.regression.base import BaseProbaRegressor
from joblib import Parallel, delayed
from sklearn.base import clone


class MultipleQuantileRegressor(BaseProbaRegressor):
    """Multiple quantile regressor.

    For quantile regression, often more than one quantile probability is of interest.
    Therefore, this multiple quantile regressor wraps multiple quantile regressors for
    multiple quantile probabilities. After fitting, all quantile regressors can be used
    to construct a quantile/interval prediction for multiple quantile probabilities.

    Parameters
    ----------
    mean_regressor : Sklearn compatible regressor
        Tabular mean regressor for predict.
    quantile_regressor : Sklearn compatible quantile regressor
        Tabular quantile regressor. In fit, for every alpha a clone of
        quantile_regressor is made whereafte the alpha parameter
        quantile_regressor_alpha_param is set using the set_params method.
        Subsequently, all regressors are fitted.
    quantile_regressor_alpha_param : str
        Parameter name that sets the quantile probability
        level for the quantile_regressor.
    alpha : list with float
        A list of probabilities. For each probability, a quantile_regressor will be fit.
    n_jobs : None or int
        The number of jobs to run in parallel for fit, predict_quantile and
        predict_interval. -1 means using all processors.
    sort_quantiles : bool
        For the quantile estimates, the lack of monotinicity can be a problem caused by
        the crossing of quantiles, also known as the quantile crossing problem. This can
        happen because each quantile is predicted by a single independent instance of
        the quantile_regressor. To resolve this issue, the quantiles for
        all predictions per row index will be sorted if sort_quantiles=True. This
        methodology will never lead to a higher quantile loss and solves the quantile
        crossing problem [1].

    [1] Victor Chernozhukov, Iván Fernández-Val, and Alfred Galichon. Quantile and
        probability curves without crossing. Econometrica, 78(3):1093–1125, 2010.

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
    ...     quantile_regressor_alpha_param="quantile",
    ...     alpha=alpha,
    ...     n_jobs=-1,
    ... )
    >>> mqr = mqr.fit(X_train, y_train)
    >>> predicted_quantiles = mqr.predict_quantiles(X_test, alpha)
    """

    _tags = {
        "capability:missing": False,
        "capability:multioutput": False,
    }

    def __init__(
        self,
        mean_regressor=None,
        quantile_regressor=None,
        quantile_regressor_alpha_param=None,
        alpha=None,
        n_jobs=None,
        sort_quantiles=False,
    ):
        self.mean_regressor = mean_regressor
        self.quantile_regressor = quantile_regressor
        self.quantile_regressor_alpha_param = quantile_regressor_alpha_param
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.sort_quantiles = sort_quantiles

        # input checks
        no_quantile_reg = False
        if (
            quantile_regressor is None
            or quantile_regressor_alpha_param is None
            or alpha is None
        ):
            no_quantile_reg = True
        elif isinstance(alpha, list):
            if len(alpha) == 0:
                no_quantile_reg = True

        no_mean_reg = False
        if mean_regressor is None:
            no_mean_reg = True

        self._no_quantile_reg = no_quantile_reg
        self._no_mean_reg = no_mean_reg
        if no_quantile_reg and no_mean_reg:
            raise ValueError(
                "Can't predict: parameters for mean and quantile "
                "regressors not provided"
            )

        super().__init__()

    def _fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame or array-like
            feature instances to fit regressor to
        y : pandas DataFrame, Series or array-like, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """

        # clone, set alpha and list all regressors
        regressors = [clone(self.mean_regressor)] if not self._no_mean_reg else []
        for a in self.alpha:
            q_est = clone(self.quantile_regressor)
            q_est = q_est.set_params(**{self.quantile_regressor_alpha_param: a})
            q_est_p = q_est.get_params()
            if q_est_p[self.quantile_regressor_alpha_param] != a:
                raise ValueError("Can't set alpha for quantile regressor.")
            else:
                regressors.append(q_est)

        # fit regressors
        def _fit_regressor(X, y, regressor):
            return regressor.fit(X=X, y=y)

        X_fit = np.array(X)
        y_fit = np.array(y)
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

        # write varname to self
        if isinstance(y, pd.DataFrame):
            # "capability:multioutput": False -> 1 column
            self._y_varname = y.columns[0]
        elif isinstance(y, pd.Series):
            self._y_varname = y.name
        else:
            self._y_varname = 0

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : pandas DataFrame or array-like, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y : pandas DataFrame, same length as `X`
            labels predicted for `X`
        """
        if self._no_mean_reg:
            raise ValueError("Can't predict: no mean_regressor provided")

        # predict
        X_pred = np.array(X)
        preds = self.regressors_["mean"].predict(X_pred)

        # format predictions as DataFrame
        preds = np.array(preds).flatten()
        preds = pd.DataFrame(preds, columns=[self._y_varname])
        if isinstance(X, (pd.DataFrame, pd.Series)):
            preds.index = X.index

        return preds

    def _predict_quantiles(self, X, alpha):
        """Compute/return quantile predictions.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and default _predict_interval

        Parameters
        ----------
        X : pandas DataFrame or array-like, must have same columns as X in `fit`
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
        if self._no_quantile_reg:
            raise ValueError(
                "Can't predict: no quantile_regressor, quantile_regressor_alpha_param "
                "or alpha provided when instantiated"
            )

        # check that we have regressors for alpha
        if not np.in1d(alpha, self.alpha).any():
            raise ValueError(
                "No regressor(s) for provided alpha. Fitted alpha: "
                f"{self.alpha}. Provided alpha: {alpha}."
            )

        # predict
        def _predict_quantile(X, regressor):
            pred = regressor.predict(X=X)
            pred = np.array(pred).flatten()
            return pred

        X_pred = np.array(X)
        regressors = [self.regressors_[a] for a in alpha]
        preds = Parallel(n_jobs=self.n_jobs)(
            delayed(_predict_quantile)(X_pred, regressor) for regressor in regressors
        )

        # format predictions as DataFrame
        preds = pd.DataFrame(np.transpose(preds))
        preds.columns = pd.MultiIndex.from_product([[self._y_varname], alpha])
        if isinstance(X, (pd.DataFrame, pd.Series)):
            preds.index = X.index

        # solve quantile crossing problem
        if self.sort_quantiles:
            preds = preds.apply(
                lambda row: row.sort_values().values, axis=1, result_type="broadcast"
            )

        return preds

    def get_params(self, deep=True):
        """Get a dict of parameters values for this object.

        Parameters
        ----------
        deep : bool, default=True
            Whether to return parameters of components.

            * If True, will return a dict of parameter name : value for this object,
              including parameters of components (= BaseObject-valued parameters).
            * If False, will return a dict of parameter name : value for this object,
              but not include parameters of components.

        Returns
        -------
        params : dict with str-valued keys
            Dictionary of parameters, paramname : paramvalue
            keys-value pairs include:

            * always: all parameters of this object, as via `get_param_names`
              values are parameter value for that key, of this object
              values are always identical to values passed at construction
            * if `deep=True`, also contains keys/value pairs of component parameters
              parameters of components are indexed as `[componentname]__[paramname]`
              all parameters of `componentname` appear as `paramname` with its value
            * if `deep=True`, also contains arbitrary levels of component recursion,
              e.g., `[componentname]__[componentcomponentname]__[paramname]`, etc
        """
        params = super().get_params(deep=deep)
        # remove quantile_regressor_alpha_param from params
        # managed by self, so removal prevents user confusion
        params.pop(f"quantile_regressor__{self.quantile_regressor_alpha_param}", None)
        return params

    def _check_X(self, X):
        # also allow array-like X
        if isinstance(X, (pd.DataFrame, pd.Series)):
            return super()._check_X(X)
        else:
            return np.array(X)

    def _check_y(self, y):
        # also allow array-like y
        if isinstance(y, (pd.DataFrame, pd.Series)):
            return super()._check_y(y)
        else:
            return np.array(y)

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
        from sklearn.linear_model import QuantileRegressor, LinearRegression
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

        params = {
            "mean_regressor": LinearRegression(),
            "quantile_regressor": QuantileRegressor(solver="highs"),
            "quantile_regressor_alpha_param": "quantile",
            "alpha": alpha,
            "n_jobs": -1,
            "sort_quantiles": True,
        }

        return params
