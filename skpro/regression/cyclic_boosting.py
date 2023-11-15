"""Cyclic boosting regressors.
This is a interface for Cyclic boosting, it contains efficient,
off-the-shelf, general-purpose supervised machine learning methods
for both regression and classification tasks.
Please read the official document for its detail
https://cyclic-boosting.readthedocs.io/en/latest/
"""

"""License.
EPL 2.0
https://github.com/Blue-Yonder-OSS/cyclic-boosting/blob/main/LICENSE
"""

__author__ = ["setoguchi-naoki"]

import numpy as np
import pandas as pd
from skpro.regression.base import BaseProbaRegressor
from skpro.distributions.qpd import J_QPD_S

# from cyclic_boosting import common_smoothers, binning
from cyclic_boosting import (
    pipeline_CBMultiplicativeQuantileRegressor,
)


# todo: change class name and write docstring
class CyclicBoosting(BaseProbaRegressor):
    """Cyclic boosting regressor.

    Estimates the parameters of Johnson Quantile-Parameterized Distributions
    (JQPD) by quantile regression, which is one of the Cyclic boosting's functions
    this method can more accurately approximate to the distribution of observed data

    Parameters
    ----------
    feature_properties : dict
        name and characteristic of train dataset
        it is able to set multiple characteristic by OR operator
        e.g. {sample1: IS_CONTINUOUS | IS_LINEAR, sample2: IS_ORDERED}
        for basic options, see
        https://cyclic-boosting.readthedocs.io/en/latest/tutorial.html#set-feature-properties
    interaction : list[tuple], default=(), optional
        some combinations of explanatory variables, (interaction term)
        e.g. [(sample1, sample2), (sample1, sample3)]
    distr_type: str, default='Normal',
        probability distribution name which is assumed for observation data's
        distribution

    Attributes
    ----------
    estimators_ : list of of skpro regressors
        clones of regressor in `estimator` fitted in the ensemble
    quantiles : list, default=[0.2, 0.5, 0.8]
        targets of quantile prediction for j-qpd's param
    quantile_values: list
        quantile prediction results
    quantile_est: list
        estimators, each estimator predicts point in the value of quantiles attribute
    qpd: skpro.distributions.J_QPD_S
        Johnson Quantile-Parameterized Distributions instance
    distr_type: str, default='QPD'
        target distribution, Option: Normal, Laplace

    Examples
    --------
    >>> from skpro.regression.cyclic_boosting import CyclicBoosting
    >>> from cyclic_boosting import flags
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> fp = {
    >>>     'age': flags.IS_CONTINUOUS,
    >>>     'sex': flags.IS_CONTINUOUS,
    >>>     'bmi': flags.IS_CONTINUOUS,
    >>>     'bp':  flags.IS_CONTINUOUS,
    >>>     's1':  flags.IS_CONTINUOUS,
    >>>     's2':  flags.IS_CONTINUOUS,
    >>>     's3':  flags.IS_CONTINUOUS,
    >>>     's4':  flags.IS_CONTINUOUS,
    >>>     's5':  flags.IS_CONTINUOUS,
    >>>     's6':  flags.IS_CONTINUOUS,
    >>> }
    >>> reg_proba = CyclicBoosting(feature_properties=fp)
    >>> reg_proba.fit(X_train, y_train)
    >>> y_pred = reg_proba.predict_proba(X_test)
    """

    _tags = {
        "object_type": "distribution",
        "estimator_type": "regressor_proba",
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, feature_properties, interaction=tuple(), distr_type="Normal"):
        self.feature_properties = feature_properties
        self.interaction = interaction
        self.quantiles = [0.2, 0.5, 0.8]
        self.quantile_values = []
        self.quantile_est = []
        self.qpd = None
        self.distr_type = distr_type

        super().__init__()

        # check parameters
        if not isinstance(feature_properties, dict):
            raise ValueError("feature_properties must be dict")
        for i in interaction:
            if not isinstance(i, tuple):
                raise ValueError("interaction must be tuple")

        # build estimators
        features = list(self.feature_properties.keys())
        for i in interaction:
            features.append(i)

        for quantile in self.quantiles:
            self.quantile_est.append(
                pipeline_CBMultiplicativeQuantileRegressor(
                    quantile=quantile,
                    feature_properties=self.feature_properties,
                    feature_groups=features,
                    maximal_iterations=50,
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

        self._y_cols = y.columns
        y = y.to_numpy().flatten()

        # multiple quantile regression for full probability estimation
        for est in self.quantile_est:
            est.fit(X.copy(), y)

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

        index = X.index
        y_cols = self._y_cols

        # quantile prediction
        for est in self.quantile_est:
            yhat = est.predict(X.copy())
            self.quantile_values.append(yhat)

        j_qpd = J_QPD_S(
            0.2,
            qv_low=self.quantile_values[0],
            qv_median=self.quantile_values[1],
            qv_high=self.quantile_values[2],
            index=index,
            columns=y_cols,
        )
        self.qpd = j_qpd

        y_pred = pd.DataFrame(j_qpd.mean(), index=index, columns=y_cols)
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
        y_pred : skpro BaseDistribution, same length as `X`
            labels predicted for `X`
        """

        # predict j-qpd
        _ = self._predict(X)

        if self.distr_type == "QPD":
            return self.qpd
        else:
            if self.distr_type == "Normal":
                from skpro.distributions.normal import Normal

                distr_type = Normal
                distr_loc_scale_name = ("mu", "sigma")
                mean_jqpd = self.qpd.mean()
                var_jqpd = np.sqrt(self.qpd.var())
            elif distr_type == "Laplace":
                from skpro.distributions.laplace import Laplace

                distr_type = Laplace
                distr_loc_scale_name = ("mu", "scale")
                mean_jqpd = self.qpd.mean()
                var_jqpd = np.sqrt(self.qpd.var()) / 2
            # elif distr_type in ["Cauchy", "t"]:
            #     from skpro.distributions.t import TDistribution

            #     distr_type = TDistribution
            #     distr_loc_scale_name = ("mu", "sigma")
            else:
                raise NotImplementedError(f"{self.distr_type} is not support")

            params = {}
            # row/column index
            ix = {"index": X.index, "columns": self._y_cols}
            params.update(ix)
            # location and scale
            loc_scale = {
                distr_loc_scale_name[0]: mean_jqpd,
                distr_loc_scale_name[1]: var_jqpd,
            }
            params.update(loc_scale)

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
        from cyclic_boosting import flags

        # NOTE: This test is only corresponded diabeat dataset
        fp = {
            "age": flags.IS_CONTINUOUS,
            "sex": flags.IS_CONTINUOUS,
            "bmi": flags.IS_CONTINUOUS,
            "bp": flags.IS_CONTINUOUS,
            "s1": flags.IS_CONTINUOUS,
            "s2": flags.IS_CONTINUOUS,
            "s3": flags.IS_CONTINUOUS,
            "s4": flags.IS_CONTINUOUS,
            "s5": flags.IS_CONTINUOUS,
            "s6": flags.IS_CONTINUOUS,
        }
        param1 = {"feature_properties": fp}
        param2 = {"feature_properties": fp, "interaction": [("age", "sex"), ("s1, s3")]}
        param3 = {
            "feature_properties": fp,
            "interaction": [("age", "sex")],
            "distr_type": "Laplace",
        }
        param4 = {
            "feature_properties": fp,
            "interaction": [("age", "sex")],
            "distr_type": "Normal",
        }
        param5 = {
            "feature_properties": fp,
            "interaction": [("age", "sex")],
            "distr_type": "QPD",
        }

        return [param1, param2, param3, param4, param5]
