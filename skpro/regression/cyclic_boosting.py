"""Cyclic boosting regressors.

This is a interface for Cyclic boosting, it contains efficient,
off-the-shelf, general-purpose supervised machine learning methods
for both regression and classification tasks.
Please read the official document for its detail
https://cyclic-boosting.readthedocs.io/en/latest/
"""

# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = [
    "setoguchi-naoki"
]  # interface only. Cyclic boosting authors in cyclic_boosting package

import warnings
from typing import Union

import numpy as np
import pandas as pd

from skpro.distributions.qpd import QPD_Johnson
from skpro.regression.base import BaseProbaRegressor


class CyclicBoosting(BaseProbaRegressor):
    """Cyclic boosting regressor from ``cyclic-boosting`` library.

    Direct interface to ``pipeline_CBAdditiveQuantileRegressor``
    and ``pipeline_CBMultiplicativeQuantileRegressor`` from ``cyclic-boosting``.

    The algorithms use boosting to create conditional distribution predictions
    that are Johnson Quantile-Parameterized Distributions (JQPD),
    with parameters estimated by quantile regression at quantile nodes.

    The quantile nodes are ``[alpha, 0.5, 1-alpha]``, where ``alpha``
    is a parameter of the model.

    The cyclic boosting model performs boosted quantile regression for the quantiles
    at the nodes, and then substitutes the quantile predictions into the paramtric
    form of the Johnson QPD.

    The model allows to select unbounded, left semi-bounded, and bounded
    predictive distribution support.

    Parameters
    ----------
    feature_groups : list, default=None
        Explanatory variables and interaction terms in the model,
        For each feature or feature tuple in the sequence, a
        one- or multidimensional factor profile will be determined,
        respectively, and used in the prediction.
        e.g. [sample1, sample2, sample3, (sample1, sample2)]
        see https://cyclic-boosting.readthedocs.io/en/latest/tutorial.html#set-features
    feature_properties : dict, default=None
        name and characteristic of train dataset by `flags` from cyclic boosting library
        it is able to set multiple characteristic by OR operator
        e.g. {sample1: flags.IS_CONTINUOUS | flags.IS_LINEAR, sample2: flags.IS_ORDERED}
        for basic options, see https://cyclic-boosting.readthedocs.io/en/latest/\
        tutorial.html#set-feature-properties
    alpha : float, default=0.2
        lower quantile QPD parameter.
        The three quantile nodes are uniquely determined by this parameter,
        as ``[alpha, 0.5, 1-alpha]``.
    mode : str, default='multiplicative'
        the type of quantile regressor. 'multiplicative' or 'additive'
    bound : str, default='U', one of ``'S'``, ``'B'``, ``'U'``
        Mode for the predictive distribution support, options are ``S``
        (semi-bounded), ``B`` (bounded), and ``U`` (unbounded).
    lower : float, default=None
        lower bound of predictive distribution support.
        If ``None`` (default), ``upper`` should also be ``None``, and the
        predictive distribution will have unbounded support, i.e., the entire reals.
        If a float, and ``upper`` is ``None``, prediction will be of
        semi-bounded support, with support between ``lower`` and infinity.
        If a float, and ``upper`` is also a float, prediction will be on a bounded
        interval, with support between ``lower`` and ``upper``.
    upper : float, default=None
        upper bound of predictive distribution support.
        If ``None`` (default), will use semi-bounded mode if ``lower`` is a float,
        and unbounded if ``lower`` is ``None``.
        If a float, assumes that ``lower`` is also a float, and prediction will
        be on a bounded interval, with support between ``lower`` and ``upper``.
    maximal_iterations : int, default=10
        maximum number of iterations for the cyclic boosting algorithm
    dist_type: str, one of ``'normal'`` (default), ``'logistic'``
        inner base distribution to use for the Johnson QPD, i.e., before
        arcosh and similar transformations.
        Available options are ``'normal'`` (default), ``'logistic'``,
        or ``'sinhlogistic'``.
    dist_shape: float, optional, default=0.0
        parameter modifying the logistic base distribution via
        sinh/arcsinh-scaling - only relevant for ``dist_type='sinhlogistic'``

    Attributes
    ----------
    estimators_ : list of skpro regressors
        clones of regressor in `estimator` fitted in the ensemble
    quantiles : list, default=[0.2, 0.5, 0.8]
        targets of quantile prediction for j-qpd's param
    quantile_values: list
        quantile prediction results
    quantile_est: list
        estimators, each estimator predicts point in the value of quantiles attribute
    qpd: skpro.distributions.J_QPD_S
        Johnson Quantile-Parameterized Distributions instance

    Example
    -------
    >>> from skpro.regression.cyclic_boosting import CyclicBoosting
    >>> from sklearn.datasets import load_diabetes  # doctest: +SKIP
    >>> from sklearn.model_selection import train_test_split  # doctest: +SKIP
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)  # doctest: +SKIP
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)  # doctest: +SKIP

    >>> reg_proba = CyclicBoosting()  # doctest: +SKIP
    >>> reg_proba.fit(X_train, y_train)  # doctest: +SKIP
    >>> y_pred = reg_proba.predict_proba(X_test)  # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["setoguchi-naoki", "felix-wick"],
        "maintainers": ["setoguchi-naoki"],
        "estimator_type": "regressor_proba",
        "python_dependencies": "cyclic_boosting>=1.4.0",
        # estimator tags
        # --------------
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(
        self,
        feature_groups=None,
        feature_properties=None,
        alpha=0.2,
        mode="multiplicative",
        bound="deprecated",
        lower=None,
        upper=None,
        maximal_iterations=10,
        dist_type: Union[str, None] = "normal",
        dist_shape: Union[float, None] = 0.0,
    ):
        self.feature_groups = feature_groups
        self.feature_properties = feature_properties
        self.alpha = alpha
        self.mode = mode
        self.bound = bound
        self.lower = lower
        self.upper = upper
        self.maximal_iterations = maximal_iterations
        self.dist_type = dist_type
        self.dist_shape = dist_shape

        super().__init__()

        self.quantiles = [self.alpha, 0.5, 1 - self.alpha]
        self.quantile_values = list()
        self.quantile_est = list()
        self.qpd = None

        # todo 2.4.0: remove bound parameter and this deprecation warning
        if bound == "deprecated":
            warnings.warn(
                "In CyclicBoosting, the 'bound' parameter is deprecated, "
                "and will be removed in skpro version 2.4.0. "
                "To retain the current behavior, and silence this warning, "
                "do not set the 'bound' parameter "
                "and set 'lower' and 'upper' parameters instead, "
                "as follows: for unbounded mode, previously bound='U', "
                "set 'lower' and 'upper' to None; "
                "for semi-bounded mode, previously bound='S', "
                "set 'lower' to lower bound and 'upper' to None; "
                "for bounded mode, previously bound='B', "
                "set 'lower' to lower bound and 'upper' to upper bound.",
                DeprecationWarning,
                stacklevel=2,
            )

        # todo 2.4.0: remove this block
        # translate bound to lower and upper
        if lower is None and bound in ["S", "B"]:
            self._lower = 0.0
        else:
            self._lower = None
        if upper is None and bound == "B":
            self._upper = 1.0
        else:
            self._upper = upper
        # end block

        # check parameters
        if (not isinstance(feature_groups, list)) and feature_groups is not None:
            raise ValueError("feature_groups needs to be list")
        if (
            not isinstance(feature_properties, dict)
        ) and feature_properties is not None:
            raise ValueError("feature_properties needs to be dict")
        if alpha >= 0.5 or alpha <= 0.0:
            raise ValueError("alpha's range needs to be 0.0 < alpha < 0.5")

        # build estimators
        if self.mode == "multiplicative":
            from cyclic_boosting import pipeline_CBMultiplicativeQuantileRegressor

            regressor = pipeline_CBMultiplicativeQuantileRegressor
        elif self.mode == "additive":
            from cyclic_boosting import pipeline_CBAdditiveQuantileRegressor

            regressor = pipeline_CBAdditiveQuantileRegressor
        else:
            raise ValueError("mode needs to be 'multiplicative' or 'additive'")

        for quantile in self.quantiles:
            self.quantile_est.append(
                regressor(
                    quantile=quantile,
                    feature_groups=feature_groups,
                    feature_properties=feature_properties,
                    maximal_iterations=maximal_iterations,
                )
            )

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
        if self.feature_groups is not None:
            feature_names = list()
            for feature in self.feature_groups:
                if isinstance(feature, tuple):
                    for f in feature:
                        feature_names.append(f)
                else:
                    feature_names.append(feature)
            if not set(feature_names).issubset(set(X.columns)):
                raise ValueError(f"{feature} is not in X")

        self._y_cols = y.columns
        y = y.to_numpy().flatten()

        # multiple quantile regression for full probability estimation
        for est in self.quantile_est:
            est.fit(X.copy(), y)

        return self

    def _predict(self, X):
        """Predict median.

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
        if self.feature_groups is not None:
            feature_names = list()
            for feature in self.feature_groups:
                if isinstance(feature, tuple):
                    for f in feature:
                        feature_names.append(f)
                else:
                    feature_names.append(feature)
            if not set(feature_names).issubset(set(X.columns)):
                raise ValueError(f"{feature} is not in X")

        index = X.index
        y_cols = self._y_cols

        # median prediction
        median_estimator = self.quantile_est[1]
        yhat = median_estimator.predict(X.copy())

        y_pred = pd.DataFrame(yhat, index=index, columns=y_cols)
        return y_pred

    def _predict_proba(self, X):
        """Predict QPD from three predicted quantile values.

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
        y_pred : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """
        if self.feature_groups is not None:
            feature_names = list()
            for feature in self.feature_groups:
                if isinstance(feature, tuple):
                    for f in feature:
                        feature_names.append(f)
                else:
                    feature_names.append(feature)
            if not set(feature_names).issubset(set(X.columns)):
                raise ValueError(f"{feature} is not in X")

        index = X.index
        y_cols = self._y_cols

        # predict quantiles
        self.quantile_values = list()
        for est in self.quantile_est:
            yhat = est.predict(X.copy())
            self.quantile_values.append(yhat)

        # todo 2.4.0: replace self._lower and self._upper with self.lower and self.upper
        # Johnson Quantile-Parameterized Distributions
        params = {
            "alpha": self.alpha,
            "qv_low": self.quantile_values[0].reshape(-1, 1),
            "qv_median": self.quantile_values[1].reshape(-1, 1),
            "qv_high": self.quantile_values[2].reshape(-1, 1),
            "lower": self.lower,
            "upper": self.upper,
            "version": self.dist_type,
            "dist_shape": self.dist_shape,
            "index": index,
            "columns": y_cols,
        }
        qpd = QPD_Johnson(**params)

        return qpd

    def _predict_interval(self, X, coverage):
        """Compute/return interval predictions.

        private _predict_interval containing the core logic,
            called from predict_interval and default _predict_quantiles

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for
        coverage : guaranteed list of float of unique values
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from ``y`` in fit,
            second level coverage fractions for which intervals were computed,
            in the same order as in input `coverage`.
            Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is equal to row index of ``X``.
            Entries are lower/upper bounds of interval predictions,
            for var in col index, at nominal coverage in second col index,
            lower/upper depending on third col index, for the row index.
            Upper/lower interval end are equivalent to
            quantile predictions at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        if self.feature_groups is not None:
            feature_names = list()
            for feature in self.feature_groups:
                if isinstance(feature, tuple):
                    for f in feature:
                        feature_names.append(f)
                else:
                    feature_names.append(feature)
            if not set(feature_names).issubset(set(X.columns)):
                raise ValueError(f"{feature} is not in X")

        index = X.index
        y_cols = self._y_cols
        columns = pd.MultiIndex.from_product(
            [y_cols, coverage, ["lower", "upper"]],
        )

        # predict interval
        interval = pd.DataFrame(index=index)
        for c in coverage:
            alpha = [0.5 - 0.5 * float(c), 0.5 + 0.5 * float(c)]
            interval = pd.concat(
                [interval, self.predict_quantiles(X=X.copy(), alpha=alpha)], axis=1
            )
        interval.columns = columns

        return interval

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
        quantiles = alpha

        if self.feature_groups is not None:
            feature_names = list()
            for feature in self.feature_groups:
                if isinstance(feature, tuple):
                    for f in feature:
                        feature_names.append(f)
                else:
                    feature_names.append(feature)
            if not set(feature_names).issubset(set(X.columns)):
                raise ValueError(f"{feature} is not in X")

        is_given_proba = False
        warning = (
            "{} percentile doesn't trained, return QPD's quantile value, "
            "which is given by predict_proba(), "
            "if you need more plausible quantile value, "
            "please train regressor again for specified quantile estimation"
        )
        if isinstance(quantiles, list):
            for q in quantiles:
                if not (q in self.quantiles):
                    warnings.warn(warning.format(q), stacklevel=2)
                    is_given_proba = True
        elif isinstance(quantiles, float):
            if not (quantiles in self.quantiles):
                warnings.warn(warning.format(quantiles), stacklevel=2)
                is_given_proba = True
        else:
            raise ValueError("quantile needs to be float or list of floats")

        index = X.index
        y_cols = self._y_cols

        columns = pd.MultiIndex.from_product(
            [y_cols, quantiles],
        )

        # predict quantiles
        self.quantile_values = list()
        if is_given_proba:
            qpd = self.predict_proba(X.copy())
            pred = np.asarray([np.squeeze(qpd.ppf(q)) for q in quantiles]).T
            quantiles = pd.DataFrame(pred, index=X.index, columns=columns)

        else:
            for est in self.quantile_est:
                yhat = est.predict(X.copy())
                self.quantile_values.append(yhat)

            quantiles = pd.DataFrame(
                np.transpose(self.quantile_values), index=index, columns=columns
            )

        return quantiles

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
        param1 = {
            "alpha": 0.2,
            "mode": "additive",
            "lower": 0.0,
            "maximal_iterations": 5,
        }
        param2 = {
            "alpha": 0.2,
            "mode": "additive",
            "lower": 0.0,
            "upper": 1000,
            "maximal_iterations": 5,
        }
        return [param1, param2]
