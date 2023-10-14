"""Base class for probabilistic regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import numpy as np
import pandas as pd

from skpro.base import BaseEstimator
from skpro.datatypes import check_is_mtype, convert
from skpro.utils.validation._dependencies import _check_estimator_deps

# allowed input mtypes
ALLOWED_MTYPES = [
    "pd_DataFrame_Table",
    "pd_Series_Table",
    "numpy1D",
    "numpy2D",
]


class BaseProbaRegressor(BaseEstimator):
    """Base class for probabilistic supervised regressors."""

    _tags = {
        "object_type": "regressor_proba",  # type of object, e.g., "distribution"
        "estimator_type": "regressor_proba",  # type of estimator, e.g., "regressor"
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, index=None, columns=None):
        self.index = index
        self.columns = columns

        super().__init__()
        _check_estimator_deps(self)

        self._X_converter_store = {}

    def __rmul__(self, other):
        """Magic * method, return (left) concatenated Pipeline.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sklearn` transformer, must follow `sklearn` API
            otherwise, `NotImplemented` is returned

        Returns
        -------
        Pipeline object,
            concatenation of `other` (first) with `self` (last).
            not nested, contains only non-Pipeline `skpro` steps
        """
        from skpro.regression.compose._pipeline import Pipeline

        # we wrap self in a pipeline, and concatenate with the other
        #   the TransformedTargetForecaster does the rest, e.g., dispatch on other
        if hasattr(other, "transform"):
            return other * Pipeline([self])
        else:
            return NotImplemented

    def fit(self, X, y):
        """Fit regressor to training data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Changes state to "fitted" = sets is_fitted flag to True

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pd.DataFrame, must be same length as X
            labels to fit regressor to

        Returns
        -------
        self : reference to self
        """
        X, y = self._check_X_y(X, y)

        # set fitted flag to True
        self._is_fitted = True

        return self._fit(X, y)

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
        raise NotImplementedError

    def predict(self, X):
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
        X = self._check_X(X)

        y_pred = self._predict(X)
        return y_pred

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
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_proba = self._has_implementation_of("_predict_proba")

        can_do_proba = implements_interval or implements_quantiles or implements_proba

        if not can_do_proba:
            raise NotImplementedError

        pred_proba = self._predict_proba(X=X)
        pred_mean = pred_proba.mean()
        return pred_mean

    def predict_proba(self, X):
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
        X = self._check_X(X)

        y_pred = self._predict_proba(X)

        # output conversion
        y_pred = convert(
            y_pred,
            from_type=self.get_tag("X_inner_mtype"),
            to_type=self._X_input_mtype,
            as_scitype="Table",
            store=self._X_converter_store,
        )

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
        # default behaviour is implemented if one of the following three is implemented
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_var = self._has_implementation_of("_predict_var")
        can_do_proba = implements_interval or implements_quantiles or implements_var

        if not can_do_proba:
            raise NotImplementedError

        # if any of the above are implemented, predict_var will have a default
        #   we use predict_var to get scale, and predict to get location
        pred_var = self.predict_var(X=X)
        pred_std = np.sqrt(pred_var)
        pred_mean = self.predict(X=X)

        from skpro.distributions.normal import Normal

        index = pred_mean.index
        columns = pred_mean.columns
        pred_dist = Normal(mu=pred_mean, sigma=pred_std, index=index, columns=columns)

        return pred_dist

    def predict_interval(self, X=None, coverage=0.90):
        """Compute/return interval predictions.

        If coverage is iterable, multiple intervals will be calculated.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for
        coverage : float or list of float of unique values, optional (default=0.90)
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
        # check that self is fitted, if not raise exception
        self.check_is_fitted()

        # check alpha and coerce to list
        coverage = self._check_alpha(coverage, name="coverage")

        # check and convert X
        X_inner = self._check_X(X=X)

        # pass to inner _predict_interval
        pred_int = self._predict_interval(X=X_inner, coverage=coverage)
        return pred_int

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
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_proba = self._has_implementation_of("_predict_proba")
        can_do_proba = implements_quantiles or implements_proba

        if not can_do_proba:
            raise NotImplementedError

        # we default to _predict_quantiles if that is implemented or _predict_proba
        # since _predict_quantiles will default to _predict_proba if it is not
        alphas = []
        for c in coverage:
            # compute quantiles corresponding to prediction interval coverage
            #  this uses symmetric predictive intervals
            alphas.extend([0.5 - 0.5 * float(c), 0.5 + 0.5 * float(c)])

        # compute quantile predictions corresponding to upper/lower
        pred_int = self._predict_quantiles(X=X, alpha=alphas)

        # change the column labels (multiindex) to the format for intervals
        # idx returned by _predict_quantiles is
        #   2-level MultiIndex with variable names, alpha
        idx = pred_int.columns
        # variable names (unique, in same order)
        var_names = idx.get_level_values(0).unique()
        # idx returned by _predict_interval should be
        #   3-level MultiIndex with variable names, coverage, lower/upper
        int_idx = pd.MultiIndex.from_product([var_names, coverage, ["lower", "upper"]])
        pred_int.columns = int_idx

        return pred_int

    def predict_quantiles(self, X=None, alpha=None):
        """Compute/return quantile predictions.

        If alpha is iterable, multiple quantiles will be calculated.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for
        alpha : float or list of float of unique values, optional (default=[0.05, 0.95])
            A probability or list of, at which quantile predictions are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from ``y`` in fit,
            second level being the values of alpha passed to the function.
            Row index is equal to row index of ``X``.
            Entries are quantile predictions, for var in col index,
            at quantile probability in second col index, for the row index.
        """
        # check that self is fitted, if not raise exception
        self.check_is_fitted()

        # default alpha
        if alpha is None:
            alpha = [0.05, 0.95]
        # check alpha and coerce to list
        alpha = self._check_alpha(alpha, name="alpha")

        # input check and conversion for X
        X_inner = self._check_X(X=X)

        # pass to inner _predict_quantiles
        quantiles = self._predict_quantiles(X=X_inner, alpha=alpha)
        return quantiles

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
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_proba = self._has_implementation_of("_predict_proba")
        can_do_proba = implements_interval or implements_proba

        if not can_do_proba:
            raise NotImplementedError

        if implements_interval:
            pred_int = pd.DataFrame()
            for a in alpha:
                # compute quantiles corresponding to prediction interval coverage
                #  this uses symmetric predictive intervals:
                coverage = abs(1 - 2 * a)

                # compute quantile predictions corresponding to upper/lower
                pred_a = self._predict_interval(X=X, coverage=[coverage])
                pred_int = pd.concat([pred_int, pred_a], axis=1)

            # now we need to subset to lower/upper depending
            #   on whether alpha was < 0.5 or >= 0.5
            #   this formula gives the integer column indices giving lower/upper
            col_selector_int = (np.array(alpha) >= 0.5) + 2 * np.arange(len(alpha))
            col_selector_bool = np.isin(np.arange(2 * len(alpha)), col_selector_int)
            num_var = len(pred_int.columns.get_level_values(0).unique())
            col_selector_bool = np.tile(col_selector_bool, num_var)

            pred_int = pred_int.iloc[:, col_selector_bool]
            # change the column labels (multiindex) to the format for intervals
            # idx returned by _predict_interval is
            #   3-level MultiIndex with variable names, coverage, lower/upper
            idx = pred_int.columns
            # variable names (unique, in same order)
            var_names = idx.get_level_values(0).unique()
            # idx returned by _predict_quantiles should be
            #   is 2-level MultiIndex with variable names, alpha
            int_idx = pd.MultiIndex.from_product([var_names, alpha])
            pred_int.columns = int_idx

        elif implements_proba:
            pred_proba = self.predict_proba(X=X)
            pred_int = pred_proba.quantile(alpha=alpha)

        return pred_int

    def predict_var(self, X=None):
        """Compute/return variance predictions.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        pred_var : pd.DataFrame
            Column names are exactly those of ``y`` passed in ``fit``.
            Row index is equal to row index of ``X``.
            Entries are variance prediction, for var in col index.
            A variance prediction for given variable and fh index is a predicted
            variance for that variable and index, given observed data.
        """
        # check that self is fitted, if not raise exception
        self.check_is_fitted()

        # check and convert X
        X_inner = self._check_X(X=X)

        # pass to inner _predict_interval
        pred_var = self._predict_var(X=X_inner)
        return pred_var

    def _predict_var(self, X):
        """Compute/return variance predictions.

        private _predict_var containing the core logic, called from predict_var

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        pred_var : pd.DataFrame
            Column names are exactly those of ``y`` passed in ``fit``.
            Row index is equal to row index of ``X``.
            Entries are variance prediction, for var in col index.
            A variance prediction for given variable and fh index is a predicted
            variance for that variable and index, given observed data.
        """
        from scipy.stats import norm

        # default behaviour is implemented if one of the following three is implemented
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_proba = self._has_implementation_of("_predict_proba")
        can_do_proba = implements_interval or implements_quantiles or implements_proba

        if not can_do_proba:
            raise NotImplementedError

        if implements_proba:
            pred_proba = self._predict_proba(X=X)
            pred_var = pred_proba.var()
            return pred_var

        # if has one of interval/quantile predictions implemented:
        #   we get quantile prediction for first and third quartile
        #   return variance of normal distribution with that first and third quartile
        if implements_interval or implements_quantiles:
            pred_int = self._predict_interval(X=X, coverage=[0.5])
            var_names = pred_int.columns.get_level_values(0).unique()
            vars_dict = {}
            for i in var_names:
                pred_int_i = pred_int[i].copy()
                # compute inter-quartile range (IQR), as pd.Series
                iqr_i = pred_int_i.iloc[:, 1] - pred_int_i.iloc[:, 0]
                # dividing by IQR of normal gives std of normal with same IQR
                std_i = iqr_i / (2 * norm.ppf(0.75))
                # and squaring gives variance (pd.Series)
                var_i = std_i**2
                vars_dict[i] = var_i

            # put together to pd.DataFrame
            #   the indices and column names are already correct
            pred_var = pd.DataFrame(vars_dict)

        return pred_var

    def _check_X_y(self, X, y):
        X, X_metadata = self._check_X(X, return_metadata=True)
        y, y_metadata = self._check_y(y)

        len_X = X_metadata["n_instances"]
        len_y = y_metadata["n_instances"]

        # input check X vs y
        if len_X != "NA" and len_y != "NA" and not len_X == len_y:
            raise ValueError(
                f"X and y in fit of {self} must have same number of rows, "
                f"but X had {len_X} rows, and y had {len_y} rows"
            )

        return X, y

    def _check_X(self, X, return_metadata=False):
        if return_metadata:
            req_metadata = ["n_instances"]
        else:
            req_metadata = []
        # input validity check for X
        valid, msg, metadata = check_is_mtype(
            X, ALLOWED_MTYPES, "Table", return_metadata=req_metadata, var_name="X"
        )

        # update with clearer message
        if not valid:
            raise TypeError(msg)

        # convert X to X_inner_mtype
        X_inner_mtype = self.get_tag("X_inner_mtype")
        X = convert(
            obj=X,
            from_type=metadata["mtype"],
            to_type=X_inner_mtype,
            as_scitype="Table",
            store=self._X_converter_store,
        )

        # if we have seen X before, check against columns
        if hasattr(self, "_X_columns") and hasattr(X, "columns"):
            if not (X.columns == self._X_columns).all():
                raise ValueError(
                    "X in predict methods must have same columns as X in fit, "
                    f"columns in fit were {self._X_columns}, "
                    f"but in predict found X.columns = {X.columns}"
                )
        # if not, remember columns
        else:
            self._X_columns = X.columns

        # remember input mtype
        self._X_input_mtype = metadata["mtype"]

        if return_metadata:
            return X, metadata
        else:
            return X

    def _check_y(self, y):
        # input validity check for y
        valid, msg, metadata = check_is_mtype(
            y, ALLOWED_MTYPES, "Table", return_metadata=["n_instances"], var_name="y"
        )

        # update with clearer message
        if not valid:
            raise TypeError(msg)

        # convert y to y_inner_mtype
        y_inner_mtype = self.get_tag("y_inner_mtype")
        y = convert(
            obj=y,
            from_type=metadata["mtype"],
            to_type=y_inner_mtype,
            as_scitype="Table",
        )

        return y, metadata

    def _check_alpha(self, alpha, name="alpha"):
        """Check that quantile or confidence level value, or list of values, is valid.

        Checks:
        alpha must be a float, or list of float, all in the open interval (0, 1).
        values in alpha must be unique.

        Parameters
        ----------
        alpha : float, list of float
        name : str, optional, default="alpha"
            the name reference to alpha displayed in the error message

        Returns
        -------
        alpha coerced to a list, i.e.: [alpha], if alpha was a float; alpha otherwise

        Raises
        ------
        ValueError
            If alpha (float) or any value in alpha (list) is outside the range (0, 1).
            If values in alpha (list) are non-unique.
        """
        # check type
        if isinstance(alpha, list):
            if not all(isinstance(a, float) for a in alpha):
                raise ValueError(
                    "When `alpha` is passed as a list, it must be a list of floats"
                )

        elif isinstance(alpha, float):
            alpha = [alpha]  # make iterable

        # check range
        for a in alpha:
            if not 0 < a < 1:
                raise ValueError(
                    f"values in {name} must lie in the open interval (0, 1), "
                    f"but found value: {a}."
                )

        # check uniqueness
        if len(set(alpha)) < len(alpha):
            raise ValueError(f"values in {name} must be unique, but found duplicates")

        return alpha
