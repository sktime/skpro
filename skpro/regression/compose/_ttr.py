"""Implements transformed target regressor for probabilistic supervised regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["TransformedTargetRegressor"]

import pandas as pd
from sklearn import clone

from skpro.distributions.trafo import TransformedDistribution
from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class TransformedTargetRegressor(BaseProbaRegressor):
    """Transformed target regressor for probabilistic supervised regression.

    The ``TransformedTargetRegressor`` takes a regressor and a transformer,
    and applies the transformer to the target variable before fitting the regressor,
    and applies the inverse of the transformer to the predictions of the regressor.

    ``fit(X, y)`` - changes state by running ``transformer.fit_transform`` with ``X=X``,
        then runnings ``regressor.fit`` with `X` being the output of
        ``transformer.fit_transform``.

    ``predict(X)`` - result is of executing ``regressor.predict``, with `X=X`
        then applies ``transformer.inverse_transform`` to the output of
        ``regressor.predict``.

    ``predict_interval(X)``, ``predict_quantiles(X)`` - as ``predict(X)``,
        with ``predict_interval`` or ``predict_quantiles`` substituted for ``predict``

    ``predict_proba(X)`` - first executes ``regressor.predict_proba(X)``,
        then returns a ``TransformedDistribution`` object with
        ``distribution=regressor.predict_proba(X)``, and
        ``transform=transformer.inverse_transform``.

    Parameters
    ----------
    regressor : skpro regressor, BaseProbaRegressor descendant
        probabilistic regressor to fit on transformed target variable
    transformer : sklearn transformer, optional (default=None)
        transformer to apply to target variable before fitting regressor,
        and to apply inverse transform to predictions of regressor

    Attributes
    ----------
    regressor_ : the fitted regressor, BaseProbaRegressor descendant
        clone of ``regressor``, fitted to transformed target variable
    transformer_ : the fitted transformer, sklearn transformer
        clone of ``transformer``, fitted to target variable

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.datasets import load_diabetes
    >>> from skpro.regression.residual import ResidualDouble
    >>> from skpro.regression.compose import TransformedTargetRegressor
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> y = pd.DataFrame(y)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg = TransformedTargetRegressor(
    ...     regressor=ResidualDouble.create_test_instance(),
    ...     transformer=StandardScaler()
    ... )
    >>> reg.fit(X_train, y_train)
    TransformedTargetRegressor(...)
    >>> y_pred = reg.predict_proba(X_test)
    """

    _tags = {
        "capability:multioutput": True,
        "capability:missing": True,
    }

    def __init__(self, regressor, transformer=None):
        self.regressor = regressor
        self.transformer = transformer
        super().__init__()

        self.regressor_ = regressor.clone()
        self.transformer_ = clone(transformer) if transformer else None

        tags_to_clone = [
            "capability:multioutput",
            "capability:survival",
            "capability:update",
        ]
        self.clone_tags(self.regressor_, tags_to_clone)

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

        # transform y
        if self.transformer_ is not None:
            t = self.transformer_
            yt = t.fit_transform(X=y)
            if not isinstance(yt, pd.DataFrame):
                yt = pd.DataFrame(yt, index=y.index)
        else:
            yt = y

        # fit regressor
        regressor = self.regressor_
        regressor.fit(X, yt, C=C)

        return self

    def _update(self, X, y, C=None):
        """Update regressor with a new batch of training data.

        State required:
            Requires state to be "fitted" = self.is_fitted=True

        Writes to self:
            Updates fitted model attributes ending in "_".

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
        yt = self.transformer_(y)
        self.regressor_.update(X=X, y=yt, C=C)
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
        y_pred = self.regressor_.predict(X=X)
        y_pred_it = self.transformer_.inverse_transform(y_pred)
        if not isinstance(y_pred_it, pd.DataFrame):
            y_cols = self._y_metadata["feature_names"]
            y_pred_it = pd.DataFrame(y_pred_it, index=X.index, columns=y_cols)
        else:
            y_pred_it.columns = self._y_metadata["feature_names"]
        return y_pred_it

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
        y_pred = self.regressor_.predict_quantiles(X=X, alpha=alpha)
        y_pred_it = self._get_inverse_transform_pred_int(
            transformer=self.transformer_, y=y_pred
        )
        cols = self._y_metadata["feature_names"]
        y_pred_it.columns = self._replace_column_level(y_pred_it.columns, cols)
        return y_pred_it

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
        y_pred = self.regressor_.predict_interval(X=X, coverage=coverage)
        y_pred_it = self._get_inverse_transform_pred_int(
            transformer=self.transformer_, y=y_pred
        )
        cols = self._y_metadata["feature_names"]
        y_pred_it.columns = self._replace_column_level(y_pred_it.columns, cols)
        return y_pred_it

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
        # explicitly using default - should be obtained from predict_proba
        return super()._predict_var(X=X)

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
        y_pred = self.regressor_.predict_proba(X=X)
        y_pred_it = TransformedDistribution(
            distribution=y_pred,
            transform=self.transformer_.inverse_transform,
            inverse_transform=self.transformer_.transform,
            assume_monotonic=True,
            index=X.index,
            columns=self._y_metadata["feature_names"],
        )
        return y_pred_it

    def _get_inverse_transform_pred_int(self, transformer, y):
        """Inverse transform predict outputs predict_interval and predict_quantiles.

        Parameters
        ----------
        transformer : sklearn transformer
            Transformer to inverse transform y
        y : pd.DataFrame
            Target series

        Returns
        -------
        y : pd.DataFrame
            Inverse transformed y
        """
        # if proba, we slice by quantile/coverage combination
        #   and collect the same quantile/coverage by variable
        #   then inverse transform, then concatenate
        idx = y.columns
        n = idx.nlevels
        idx_low = idx.droplevel(0).unique()
        yt = dict()
        for ix in idx_low:
            levels = list(range(1, n))
            if len(levels) == 1:
                levels = levels[0]
            yt[ix] = y.xs(ix, level=levels, axis=1)
            if len(yt[ix].columns) == 1:
                temp = yt[ix].columns
                yt[ix].columns = self._y_metadata["feature_names"]
            yt[ix] = transformer.inverse_transform(X=yt[ix])
            if not isinstance(yt[ix], pd.DataFrame):
                yt[ix] = pd.DataFrame(yt[ix], index=y.index, columns=temp)
            elif len(yt[ix].columns) == 1:
                yt[ix].columns = temp
        y = pd.concat(yt, axis=1)
        flipcols = [n - 1] + list(range(n - 1))
        y.columns = y.columns.reorder_levels(flipcols)
        y = y.loc[:, idx]

        return y

    def _replace_column_level(self, ix, new_values):
        """Replace the values at level 0 of a MultiIndex.

        Parameters
        ----------
        ix : pd.MultiIndex
            The input MultiIndex columns.
        new_values : Iterable
            New values to replace at level 0.

        Returns
        -------
        pd.MultiIndex
            A new MultiIndex with updated MultiIndex.
        """
        from collections import OrderedDict

        assert isinstance(ix, pd.MultiIndex)

        # Get level 0 values
        level_0_vals = ix.get_level_values(0)

        # Determine the unique values in order of first appearance
        unique_vals = list(OrderedDict.fromkeys(level_0_vals))
        if len(new_values) != len(unique_vals):
            raise ValueError(
                "Length of new_values must match number of unique values in level 0."
            )

        # Create mapping from old to new
        mapping = dict(zip(unique_vals, new_values))
        replaced_level_0 = [mapping[val] for val in level_0_vals]

        # Construct new levels
        new_levels = [
            replaced_level_0 if i == 0 else ix.get_level_values(i)
            for i in range(ix.nlevels)
        ]

        return pd.MultiIndex.from_arrays(new_levels)

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
        from sklearn.preprocessing import StandardScaler

        from skpro.regression.linear import DummyProbaRegressor
        from skpro.survival.compose import ConditionUncensored

        params1 = {
            "regressor": DummyProbaRegressor(),
            "transformer": StandardScaler(),
        }
        params2 = {
            "regressor": ConditionUncensored.create_test_instance(),
            "transformer": StandardScaler(),
        }
        return [params1, params2]
