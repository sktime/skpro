"""Meta-strategy for online learning: turn off online update."""

__author__ = ["fkiraly"]
__all__ = ["OnlineDontRefit"]

from skpro.regression.base import _DelegatedProbaRegressor


class OnlineDontRefit(_DelegatedProbaRegressor):
    """Simple online regression strategy, turns off any refitting.

    In ``fit``, behaves like the wrapped regressor.
    In ``update``, does nothing, overriding any other logic.

    This strategy is useful when the wrapped regressor is already an online regressor,
    to create a "no-op" online regressor for comparison.

    Parameters
    ----------
    estimator : skpro regressor, descendant of BaseProbaRegressor
        regressor to be update-refitted on all data, blueprint

    Attributes
    ----------
    estimator_ : skpro regressor, descendant of BaseProbaRegressor
        clone of the regressor passed in the constructor, fitted on all data
    """

    _tags = {"capability:update": False}

    def __init__(self, estimator):
        self.estimator = estimator

        super().__init__()

        tags_to_clone = [
            "capability:missing",
            "capability:survival",
        ]
        self.clone_tags(estimator, tags_to_clone)

        self.estimator_ = self.estimator.clone()

    def _update(self, X, y, C=None):
        """Update regressor with new batch of training data.

        State required:
            Requires state to be "fitted".

        Writes to self:
            Updates fitted model attributes ending in "_".

        Parameters
        ----------
        X : pandas DataFrame
            feature instances to fit regressor to
        y : pandas DataFrame, must be same length as X
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
        from skbase.utils.dependencies import _check_estimator_deps
        from sklearn.linear_model import LinearRegression, Ridge

        from skpro.regression.residual import ResidualDouble
        from skpro.survival.coxph import CoxPH

        regressor = ResidualDouble(LinearRegression())

        params = [{"estimator": regressor}]

        if _check_estimator_deps(CoxPH, severity="none"):
            coxph = CoxPH()
            params.append({"estimator": coxph})
        else:
            ridge = Ridge()
            params.append({"estimator": ResidualDouble(ridge)})

        return params
