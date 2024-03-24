"""Implements pipelines for probabilistic supervised regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# based on sktime pipelines

__author__ = ["fkiraly"]
__all__ = ["Pipeline"]

import pandas as pd
from sklearn import clone

from skpro.base import BaseMetaEstimator
from skpro.regression.base import BaseProbaRegressor
from skpro.utils.sklearn import prep_skl_df


class _Pipeline(BaseMetaEstimator, BaseProbaRegressor):
    """Abstract class for pipelines."""

    # for default get_params/set_params from BaseMetaEstimator
    # named_object_parameters points to the attribute of self
    # which contains the heterogeneous set of estimators
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    # if the estimator is fittable, _BaseMetaEstimator also
    # provides an override for get_fitted_params for params from the fitted estimators
    # the fitted estimators should be in fitted_named_object_parameters
    # this must be an iterable of (name: str, estimator, ...) tuples for the default
    _tags = {
        "named_object_parameters": "_steps",
        "fitted_named_object_parameters": "steps_",
    }

    def _get_pipeline_scitypes(self, estimators):
        """Get list of scityes (str) from names/estimator list."""

        def est_scitype(tpl):
            est = tpl[1]
            if isinstance(est, BaseProbaRegressor):
                return "regressor"
            elif hasattr(est, "transform"):
                return "transformer"
            else:
                return "other"

        return [est_scitype(x) for x in estimators]

    def _get_regressor_index(self, estimators):
        """Get the index of the first regressor in the list."""
        return self._get_pipeline_scitypes(estimators).index("regressor")

    def _check_steps(self, estimators, allow_postproc=False):
        """Check Steps.

        Parameters
        ----------
        estimators : list of estimators, or list of (name, estimator) pairs
        allow_postproc : bool, optional, default=False
            whether transformers after the regressor are allowed

        Returns
        -------
        step : list of (name, estimator) pairs, estimators are cloned (not references)
            if estimators was a list of (str, estimator) tuples, then just cloned
            if was a list of estimators, then str are generated via _get_estimator_names

        Raises
        ------
        TypeError if names in `estimators` are not unique
        TypeError if estimators in `estimators` are not all regressor or transformer
        TypeError if there is not exactly one regressor in `estimators`
        TypeError if not allow_postproc and regressor is not last estimator
        """
        if not isinstance(estimators, list):
            msg = (
                f"steps in {self.name} must be list of estimators, "
                f"or (string, estimator) pairs, "
                f"the two can be mixed; but, found steps of type {type(estimators)}"
            )
            raise TypeError(msg)

        estimator_tuples = self._coerce_to_named_object_tuples(estimators)
        estimator_tuples = [(name, clone(est)) for name, est in estimator_tuples]

        def _set_pd(est):
            if hasattr(est, "transform") and hasattr(est, "set_output"):
                est.set_output(transform="pandas")
            return est

        # ensure sklearn transformers produce pandas output
        estimator_tuples = [(name, _set_pd(est)) for name, est in estimator_tuples]

        names, estimators = zip(*estimator_tuples)

        # validate names
        self._check_names(names)

        scitypes = self._get_pipeline_scitypes(estimator_tuples)
        if not set(scitypes).issubset(["regressor", "transformer"]):
            raise TypeError(
                f"estimators passed to {self.name} "
                f"must be either transformer or regressor"
            )
        if scitypes.count("regressor") != 1:
            raise TypeError(
                f"exactly one regressor must be contained in the chain, "
                f"but found {scitypes.count('regressor')}"
            )

        regressor_ind = self._get_regressor_index(estimator_tuples)

        if not allow_postproc and regressor_ind != len(estimators) - 1:
            TypeError(
                f"in {self.name}, last estimator must be a regressor, "
                f"but found a transformer"
            )

        # Shallow copy
        return estimator_tuples

    def _iter_transformers(self, reverse=False, fc_idx=-1):
        # exclude final regressor
        steps = self.steps_[:fc_idx]

        if reverse:
            steps = reversed(steps)

        for idx, (name, transformer) in enumerate(steps):
            yield idx, name, transformer

    def __len__(self):
        """Return the length of the Pipeline."""
        return len(self.steps)

    @property
    def named_steps(self):
        """Map the steps to a dictionary."""
        return dict(self._steps)

    @property
    def _steps(self):
        return self._coerce_to_named_object_tuples(self.steps, clone=False)

    @_steps.setter
    def _steps(self, value):
        self.steps = value

    def _components(self, base_class=None):
        """Return references to all state changing BaseObject type attributes.

        This *excludes* the blue-print-like components passed in the __init__.

        Caution: this method returns *references* and not *copies*.
            Writing to the reference will change the respective attribute of self.

        Parameters
        ----------
        base_class : class, optional, default=None, must be subclass of BaseObject
            if None, behaves the same as `base_class=BaseObject`
            if not None, return dict collects descendants of `base_class`

        Returns
        -------
        dict with key = attribute name, value = reference to attribute
        dict contains all attributes of `self` that inherit from `base_class`, and:
            whose names do not contain the string "__", e.g., hidden attributes
            are not class attributes, and are not hyper-parameters (`__init__` args)
        """
        import inspect

        from skpro.base import BaseEstimator

        if base_class is None:
            base_class = BaseEstimator
        if base_class is not None and not inspect.isclass(base_class):
            raise TypeError(f"base_class must be a class, but found {type(base_class)}")
        # if base_class is not None and not issubclass(base_class, BaseObject):
        #     raise TypeError("base_class must be a subclass of BaseObject")

        fitted_estimator_tuples = self.steps_

        comp_dict = {name: comp for (name, comp) in fitted_estimator_tuples}
        return comp_dict

    # both children use the same step params for testing, so putting it here
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

        from skpro.regression.residual import ResidualDouble
        from skpro.survival.coxph import CoxPH
        from skpro.utils.validation._dependencies import _check_estimator_deps

        params = []

        regressor = ResidualDouble.create_test_instance()

        STEPS1 = [
            ("transformer", StandardScaler()),
            ("regressor", regressor),
        ]
        params += [{"steps": STEPS1}]

        params += [{"steps": [StandardScaler(), regressor]}]

        # testing with survival predictor
        if _check_estimator_deps(CoxPH, severity="none"):
            params += [{"steps": [StandardScaler(), CoxPH()]}]

        return params


class Pipeline(_Pipeline):
    """Pipeline for probabilistic supervised regression.

    Pipeline is only applying the given transformers
    to X. The regressor can also be a TransformedTargetregressor containing
    transformers to transform y.

    For a list `t1`, `t2`, ..., `tN`, `r`
        where `t[i]` are transformers, and `r` is an sktime regressor,
        the pipeline behaves as follows:

    `fit(X, y)` - changes state by running `t1.fit_transform` with `X=X`, `y=y`
        then `t2.fit_transform` on `X=` the output of `t1.fit_transform`, `y=y`, etc
        sequentially, with `t[i]` receiving the output of `t[i-1]` as `X`,
        then running `r.fit` with `X` being the output of `t[N]`, and `y=y`
    `predict(X)` - result is of executing `r.predict`, with `X=X`
        being the result of the following process:
        running `t1.fit_transform` with `X=X`,
        then `t2.fit_transform` on `X=` the output of `t1.fit_transform`, etc
        sequentially, with `t[i]` receiving the output of `t[i-1]` as `X`,
        and returning th output of `tN` to pass to `r.predict` as `X`.
    `predict_interval(X)`, `predict_quantiles(X)` - as `predict(X)`,
        with `predict_interval` or `predict_quantiles` substituted for `predict`
    `predict_var`, `predict_proba` - uses base class default to obtain
        crude estimates from `predict_quantiles`.

    `get_params`, `set_params` uses `sklearn` compatible nesting interface
        if list is unnamed, names are generated as names of classes
        if names are non-unique, `f"_{str(i)}"` is appended to each name string
            where `i` is the total count of occurrence of a non-unique string
            inside the list of names leading up to it (inclusive)

    `Pipeline` can also be created by using the magic multiplication
        on any regressor, i.e., if `my_regressor` inherits from `BaseProbaRegressor`,
            and `my_t1`, `my_t2`, are an `sklearn` transformer,
            then, for instance, `my_t1 * my_t2 * my_regressor`
            will result in the same object as  obtained from the constructor
            `Pipeline([my_t1, my_t2, my_regressor])`
        magic multiplication can also be used with (str, transformer) pairs,
            as long as one element in the chain is a regressor

    Parameters
    ----------
    steps : list of sktime transformers and regressors, or
        list of tuples (str, estimator) of sktime transformers or regressors
            the list must contain exactly one regressor
        these are "blueprint" transformers resp regressors,
            regressor/transformer states do not change when `fit` is called

    Attributes
    ----------
    steps_ : list of tuples (str, estimator) of sktime transformers or regressors
        clones of estimators in `steps` which are fitted in the pipeline
        is always in (str, estimator) format, even if `steps` is just a list
        strings not passed in `steps` are replaced by unique generated strings
        i-th transformer in `steps_` is clone of i-th in `steps`
    regressor_ : estimator, reference to the unique regressor in steps_

    Examples
    --------
    >>> from skpro.regression.compose import Pipeline
    >>> from skpro.regression.residual import ResidualDouble
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.impute import SimpleImputer as Imputer
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.preprocessing import MinMaxScaler
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg_mean = LinearRegression()
    >>> reg_proba = ResidualDouble(reg_mean)

    Example 1: string/estimator pairs

    >>> pipe = Pipeline(steps=[
    ...     ("imputer", Imputer()),
    ...     ("scaler", MinMaxScaler()),
    ...     ("regressor", reg_proba),
    ... ])
    >>> pipe.fit(X_train, y_train)
    Pipeline(...)
    >>> y_pred = pipe.predict(X=X_test)
    >>> y_pred_proba = pipe.predict_proba(X=X_test)

    Example 2: without strings

    >>> pipe = Pipeline([
    ...     Imputer(),
    ...     MinMaxScaler(),
    ...     ("regressor", reg_proba),
    ... ])

    Example 3: using the dunder method
    (requires bracketing as sklearn does not support dunders)

    >>> reg_proba = ResidualDouble(reg_mean)
    >>> pipe = Imputer() * (MinMaxScaler() * reg_proba)
    """

    _tags = {
        "capability:multioutput": True,
        "capability:missing": True,
    }

    def __init__(self, steps):
        self.steps = steps
        self.steps_ = self._check_steps(steps, allow_postproc=False)

        super().__init__()

        tags_to_clone = ["capability:multioutput", "capability:survival"]
        self.clone_tags(self.regressor_, tags_to_clone)

    @property
    def regressor_(self):
        """Return reference to the regressor in the pipeline.

        Valid after _fit.
        """
        return self.steps_[-1][1]

    def __rmul__(self, other):
        """Magic * method, return (left) concatenated Pipeline.

        Implemented for `other` being a transformer, otherwise returns `NotImplemented`.

        Parameters
        ----------
        other: `sktime` transformer, must inherit from BaseTransformer
            otherwise, `NotImplemented` is returned

        Returns
        -------
        Pipeline object,
            concatenation of `other` (first) with `self` (last).
            not nested, contains only non-TransformerPipeline `sktime` steps
        """
        _, ests = zip(*self.steps_)
        names = tuple(self._get_names_and_objects(self.steps)[0])

        if hasattr(other, "transform"):
            new_names = (type(other).__name__,) + names
            new_ests = (other,) + ests
        elif isinstance(other, tuple) and len(other) == 2:
            other_name = other[0]
            other_trafo = other[1]
            new_names = (other_name,) + names
            new_ests = (other_trafo,) + ests
        else:
            return NotImplemented

        # if all the names are equal to class names, we eat them away
        if all(type(x[1]).__name__ == x[0] for x in zip(new_names, new_ests)):
            return Pipeline(steps=list(new_ests))
        else:
            return Pipeline(steps=list(zip(new_names, new_ests)))

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

        # transform X
        for step_idx, name, transformer in self._iter_transformers():
            t = transformer
            X = t.fit_transform(X=X, y=y)
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, index=y.index)
            self.steps_[step_idx] = (name, t)

        # fit regressor
        name, regressor = self.steps_[-1]
        r = regressor.clone()
        r.fit(X, y, C=C)
        self.steps_[-1] = (name, r)

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
        X = self._transform(X)
        return self.regressor_.predict(X=X)

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
        X = self._transform(X)
        return self.regressor_.predict_quantiles(X=X, alpha=alpha)

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
        X = self._transform(X)
        return self.regressor_.predict_interval(X=X, coverage=coverage)

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
        X = self._transform(X)
        return self.regressor_.predict_var(X=X)

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
        X = self._transform(X)
        return self.regressor_.predict_proba(X=X)

    def _transform(self, X, y=None):
        """Transform data."""
        # coerce X to pandas DataFrame with string column names
        X = prep_skl_df(X, copy_df=True)

        for _, _, transformer in self._iter_transformers():
            if self._has_y_arg(transformer.transform):
                Xt = transformer.transform(X=X, y=y)
            else:
                Xt = transformer.transform(X=X)
            if not isinstance(Xt, pd.DataFrame):
                Xt = pd.DataFrame(Xt, index=X.index)
        return Xt

    def _has_y_arg(self, method):
        """Check if method has y argument."""
        from inspect import signature

        sig = signature(method)
        return "y" in sig.parameters
