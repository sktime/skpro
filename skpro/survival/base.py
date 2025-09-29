"""Base class for probabilistic survival regression."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.regression.base import BaseProbaRegressor

# allowed input mtypes
ALLOWED_MTYPES = [
    "pd_DataFrame_Table",
    "pd_Series_Table",
    "numpy1D",
    "numpy2D",
]


class BaseSurvReg(BaseProbaRegressor):
    """Base class for survival regression models.

    Contains no additional logic, only docstring overrides.
    """

    _tags = {"capability:survival": True}

    def fit(self, X, y, C=None):
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

            * should have same column name as y, same length as X and y
            * should have entries 0 and 1 (float or int),
            0 = uncensored, 1 = (right) censored

            if None, all observations are assumed to be uncensored.

        Returns
        -------
        self : reference to self
        """
        super().fit(X=X, y=y, C=C)
        return self

    def update(self, X, y, C=None):
        """Update regressor with a new batch of training data.

        Only estimators with the ``capability:update`` tag (value ``True``)
        provide this method, otherwise the method ignores the call and
        discards the data passed.

        State required:
            Requires state to be "fitted".

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

        Returns
        -------
        self : reference to self
        """
        super().update(X=X, y=y, C=C)
        return self
