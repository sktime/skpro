"""Reducer to supervised regression - fit on uncensored."""

__author__ = ["fkiraly"]

from skpro.regression.base._delegate import _DelegatedProbaRegressor


class FitUncensored(_DelegatedProbaRegressor):
    """Reduction to tabular probabilistic regression - fit on uncensored subsample.

    Simple baseline reduction strategy for predictive survival analysis,
    fits a probabilistic regressor on the uncensored subsample of the data.

    In ``fit``, drops all observations that are right censored,
    and fits a probabilistic regressor on the remaining data.

    In ``predict``, uses the ``predict`` method of the fitted regressor.

    Parameters
    ----------
    estimator : skpro regressor, BaseProbaRegressor descendant
        probabilistic regressor to fit on uncensored subsample

    Attributes
    ----------
    estimator_ : skpro regressor, BaseProbaRegressor descendant
        fitted probabilistic regressor, fit on uncensored subsample
        clone of estimator
    """

    _tags = {"capability:survival": True}

    _delegate_name = "estimator_"

    def __init__(self, estimator):
        self.estimator = estimator

        super().__init__()

    def _fit(self, X, y, C=None):
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
        C : pd.DataFrame, optional (default=None)
            censoring information for survival analysis,
            should have same column name as y, same length as X and y
            should have entries 0 and 1 (float or int)
            0 = uncensored, 1 = (right) censored
            if None, all observations are assumed to be uncensored
            Can be passed to any probabilistic regressor,
            but is ignored if capability:survival tag is False.
        """
        if C is None:
            self.estimator_ = self.estimator.clone()
            self.estimator_.fit(X, y)
            return self

        uncensored_index = C.index[C.iloc[:, 0] == 0].tolist()

        X_uncensored = X.loc[uncensored_index, :]
        y_uncensored = y.loc[uncensored_index, :]

        self.estimator_ = self.estimator.clone()
        self.estimator_.fit(X_uncensored, y_uncensored)

        return self

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
        from skpro.regression.bootstrap import BootstrapRegressor
        from skpro.regression.residual import ResidualDouble

        param1 = {"estimator": ResidualDouble.create_test_instance()}
        param2 = {"estimator": BootstrapRegressor.create_test_instance()}

        return [param1, param2]
