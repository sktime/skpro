"""Extension template for regressors."""
# todo: write an informative docstring for the file or module, remove the above
# todo: add an appropriate copyright notice for your estimator
#       estimators contributed to skpro should have the copyright notice at the top
#       estimators of your own do not need to have permissive or BSD-3 copyright

# todo: uncomment the following line, enter authors' GitHub IDs
# __author__ = [authorGitHubID, anotherAuthorGitHubID]

from skpro.regression.base import BaseProbaRegressor

# todo: add any necessary imports here

# todo: if any imports are skpro soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name


# todo: change class name and write docstring
class ClassName(BaseProbaRegressor):
    """Custom probabilistic supervised regressor. todo: write docstring.

    todo: describe your custom regressor here

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
    and so on
    est : skpro.estimator, BaseEstimator descendant
        descriptive explanation of est
    est2: another estimator
        descriptive explanation of est2
    and so on
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    # tags inherited from base are "safe defaults" which can usually be left as-is
    _tags = {
        # packaging info
        # --------------
        "authors": ["author1", "author2"],  # authors, GitHub handles
        "maintainers": ["maintainer1", "maintainer2"],  # maintainers, GitHub handles
        # author = significant contribution to code at some point
        # maintainer = algorithm maintainer role, "owner"
        # specify one or multiple authors and maintainers, only for skpro contribution
        # remove maintainer tag if maintained by skpro/sktim core team
        #
        "python_version": None,  # PEP 440 python version specifier to limit versions
        "python_dependencies": None,  # PEP 440 python dependencies specifier,
        # e.g., "numba>0.53", or a list, e.g., ["numba>0.53", "numpy>=1.19.0"]
        # delete if no python dependencies or version limitations
        #
        # estimator tags
        # --------------
        "capability:multioutput": False,  # can the estimator handle multi-output data?
        "capability:missing": True,  # can the estimator handle missing data?
        "X_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit, _predict
        "y_inner_mtype": "pd_DataFrame_Table",  # type seen in internal _fit
    }

    # todo: fill init
    # params should be written to self and never changed
    # super call must not be removed, change class name
    # parameter checks can go after super call
    def __init__(self, paramname, paramname2="paramname2default"):
        # estimators should precede parameters
        #  if estimators have default values, set None and initalize below

        # todo: write any hyper-parameters and components to self
        self.paramname = paramname
        self.paramname2 = "paramname2default"

        # leave this as is
        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.parama etc
        # instead, write to self._parama, self._newparam (starting with _)

        # todo: default estimators should have None arg defaults
        #  and be initialized here
        #  do this only with default estimators, not with parameters
        # if est2 is None:
        #     self.estimator = MyDefaultEstimator()

        # todo: if tags of estimator depend on component tags, set these here
        #  only needed if estimator is a composite
        #  tags set in the constructor apply to the object and override the class
        #
        # example 1: conditional setting of a tag
        # if est.foo == 42:
        #   self.set_tags(handles-missing-data=True)
        # example 2: cloning tags from component
        #   self.clone_tags(est2, ["enforce_index_type", "handles-missing-data"])

    # todo: implement this, mandatory
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
        # insert logic for estimator here
        # fitted parameters should be written to parameters ending in underscore

        # self must be returned at the end
        return self

    # todo: implement this, mandatory
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
        # implement logic for prediction here
        # this can read out parameters fitted in fit, or hyperparameters from init
        # no attributes should be written to self

        y_pred = "placeholder"
        # returned object should be pd.DataFrame
        # same length as X, same columns as y in fit
        return y_pred

    # todo: implement at least one of the probabilistic prediction methods
    # _predict_proba, _predict_interval, _predict_quantiles
    # if one is implemented, the other two are filled in by default
    # implementation of _predict_proba is preferred, if possible
    #
    # CAVEAT: if not implemented, _predict_proba assumes normal distribution
    # this can be inconsistent with _predict_interval or _predict_quantiles
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
        # if implementing _predict_proba (otherwise delete this method)
        # todo: adapt the following by filling in logic to produce prediction values

        # boilerplate code to create correct output index
        index = X.index
        y_cols = self._y_cols  # columns from y in fit, not automatically stored
        columns = y_cols

        # values = logic to produce prediction values
        # replace this import by the distribution you are using
        # the distribution type can be conditional, e.g., data or parameter dependent
        from skpro.distributions import SomeDistribution

        values = None  # fill in values
        y_pred = SomeDistribution(values, index=index, columns=columns)

        return y_pred

    # todo: implement at least one of the probabilistic prediction methods, see above
    # delete the methods that are not implemented and filled by default
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
        # if implementing _predict_interval (otherwise delete this method)
        # todo: adapt the following by filling in logic to produce prediction values

        # boilerplate code to create correct pandas output index
        # only if using pandas, for other mtypes, use appropriate data structure
        import pandas as pd

        index = X.index
        y_cols = self._y_cols  # columns from y in fit, not automatically stored
        columns = pd.MultiIndex.from_product(
            [y_cols, coverage, ["lower", "upper"]],
        )

        # values = logic to produce prediction values
        values = None  # fill in values
        pred_int = pd.DataFrame(values, index=index, columns=columns)

        return pred_int

    # todo: implement at least one of the probabilistic prediction methods, see above
    # delete the methods that are not implemented and filled by default
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
        # if implementing _predict_quantiles (otherwise delete this method)
        # todo: adapt the following by filling in logic to produce prediction values

        # boilerplate code to create correct pandas output index
        # only if using pandas, for other mtypes, use appropriate data structure
        import pandas as pd

        index = X.index
        y_cols = self._y_cols  # columns from y in fit, not automatically stored
        columns = pd.MultiIndex.from_product(
            [y_cols, alpha],
        )

        # values = logic to produce prediction values
        values = None  # fill in values
        quantiles = pd.DataFrame(values, index=index, columns=columns)

        return quantiles

    # todo: return default parameters, so that a test instance can be created
    #   required for automated unit and integration testing of estimator
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

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        #
        # this can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from skpro or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for most automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        #   For classification, this is also used in tests for reference settings,
        #       such as published in benchmarking studies, or for identity testing.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params
        # params = {"est": value3, "parama": value4}
        # return params
