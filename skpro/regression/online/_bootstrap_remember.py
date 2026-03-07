"""Meta-strategy for online learning: remember bootstrap sample, pool with batches."""

__all__ = ["OnlineBootstrapRemember"]

import pandas as pd
from sklearn.utils import check_random_state

from skpro.distributions.mixture import Mixture
from skpro.regression.base import _DelegatedProbaRegressor


class OnlineBootstrapRemember(_DelegatedProbaRegressor):
    """Online regression strategy using bootstrap sample memory.

    Remembers a smaller bootstrap sample of size `n_remember` from the fit data,
    and pools this with data in new batches. At each update, bootstraps so the
    remembered sample stays `n_remember` size.

    Predicts using a Mixture of distributions from:
    1. Regressor fitted on remembered sample only
    2. Regressor fitted on pooled data (remembered + new batch)

    Weights are proportional to sample sizes.

    Parameters
    ----------
    estimator : skpro regressor, descendant of BaseProbaRegressor
        regressor to be fitted, blueprint
    n_remember : int, default=100
        Size of bootstrap sample to remember from fit data
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    Attributes
    ----------
    estimator_remembered_ : skpro regressor
        regressor fitted on remembered bootstrap sample
    estimator_pooled_ : skpro regressor
        regressor fitted on pooled data (remembered + latest batch)
    """

    _tags = {"capability:update": True}

    def __init__(self, estimator, n_remember=100, random_state=None):
        self.estimator = estimator
        self.n_remember = n_remember
        self.random_state = random_state
        self._random_state = check_random_state(random_state)

        super().__init__()

        tags_to_clone = [
            "capability:missing",
            "capability:survival",
        ]
        self.clone_tags(estimator, tags_to_clone)

    def _fit(self, X, y, C=None):
        """Fit regressor to training data.

        Fits regressor on full data, then creates bootstrap sample of size
        n_remember to remember.

        Writes to self:
            Sets fitted model attributes ending in "_".

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
        n_samples = len(X)

        estimator_full = self.estimator.clone()
        estimator_full.fit(X=X, y=y, C=C)

        n_remember = min(self.n_remember, n_samples)
        bootstrap_indices = self._random_state.choice(
            n_samples, size=n_remember, replace=True
        )

        X_remembered = X.iloc[bootstrap_indices].reset_index(drop=True)
        y_remembered = y.iloc[bootstrap_indices].reset_index(drop=True)
        if C is not None:
            C_remembered = C.iloc[bootstrap_indices].reset_index(drop=True)
        else:
            C_remembered = None

        estimator_remembered = self.estimator.clone()
        estimator_remembered.fit(X=X_remembered, y=y_remembered, C=C_remembered)

        self.estimator_remembered_ = estimator_remembered
        self.estimator_pooled_ = estimator_full
        self._remembered_X = X_remembered
        self._remembered_y = y_remembered
        self._remembered_C = C_remembered
        self._n_remembered = n_remember
        self._n_pooled = n_samples
        self._n_new = 0

        return self

    def _update_data(self, X, X_new):
        """Update data with new batch of training data.

        Treats X_new as data with new indices, even if some indices overlap with X.

        Parameters
        ----------
        X : pandas DataFrame
        X_new : pandas DataFrame

        Returns
        -------
        X_updated : pandas DataFrame
            concatenated data, with reset index
        """
        if X is None and X_new is None:
            return None
        if X is None and X_new is not None:
            return X_new.reset_index(drop=True)
        if X is not None and X_new is None:
            return X.reset_index(drop=True)
        # else, both X and X_new are not None
        X_updated = pd.concat([X, X_new], ignore_index=True)
        return X_updated

    def _predict(self, X):
        """Predict labels for data from features.

        Returns the mean of the Mixture distribution.

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
        pred_proba = self._predict_proba(X)
        return pred_proba.mean()

    def _predict_quantiles(self, X, alpha):
        """Compute/return quantile predictions.

        Uses the Mixture distribution's quantile method.

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
        pred_proba = self._predict_proba(X)
        return pred_proba.quantile(alpha)

    def _predict_interval(self, X, coverage):
        """Compute/return interval predictions.

        Uses the Mixture distribution's quantile method.

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
        """
        import pandas as pd

        pred_proba = self._predict_proba(X)
        pred_int = pd.DataFrame(index=X.index)

        for c in coverage:
            alpha_lower = 0.5 - 0.5 * float(c)
            alpha_upper = 0.5 + 0.5 * float(c)
            quantiles = pred_proba.quantile([alpha_lower, alpha_upper])

            varnames = self._get_varnames()
            for var in varnames:
                lower = quantiles[(var, alpha_lower)]
                upper = quantiles[(var, alpha_upper)]
                pred_int[(var, c, "lower")] = lower
                pred_int[(var, c, "upper")] = upper

        int_idx = self._get_columns(method="predict_interval", coverage=coverage)
        pred_int.columns = int_idx

        return pred_int

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        Returns a Mixture distribution combining predictions from:
        1. Regressor fitted on remembered sample only
        2. Regressor fitted on pooled data (remembered + latest batch)

        Weights are proportional to sample sizes.

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
        y : skpro Mixture distribution, same length as `X`
            labels predicted for `X`
        """
        pred_remembered = self.estimator_remembered_.predict_proba(X)
        pred_pooled = self.estimator_pooled_.predict_proba(X)

        distributions = [
            ("remembered", pred_remembered),
            ("pooled", pred_pooled),
        ]

        if hasattr(self, "_n_new") and self._n_new > 0:
            weights = [self._n_remembered, self._n_new]
        elif hasattr(self, "_n_pooled"):
            n_original = self._n_pooled - self._n_remembered
            if n_original > 0:
                weights = [self._n_remembered, n_original]
            else:
                weights = [self._n_remembered, self._n_remembered]
        else:
            weights = [self._n_remembered, self._n_remembered]

        return Mixture(distributions=distributions, weights=weights)

    def _predict_var(self, X):
        """Compute/return variance predictions.

        Uses the Mixture distribution's variance method.

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
        """
        pred_proba = self._predict_proba(X)
        return pred_proba.var()

    def _update(self, X, y, C=None):
        """Update regressor with new batch of training data.

        Pools remembered sample with new batch, fits regressor on pooled data,
        then bootstraps new remembered sample of size n_remember from pooled data.

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
        X_pooled = self._update_data(self._remembered_X, X)
        y_pooled = self._update_data(self._remembered_y, y)
        C_pooled = self._update_data(self._remembered_C, C)

        n_pooled = len(X_pooled)
        n_new = len(X)

        estimator_pooled = self.estimator.clone()
        estimator_pooled.fit(X=X_pooled, y=y_pooled, C=C_pooled)

        n_remember = min(self.n_remember, n_pooled)
        bootstrap_indices = self._random_state.choice(
            n_pooled, size=n_remember, replace=True
        )

        X_remembered = X_pooled.iloc[bootstrap_indices].reset_index(drop=True)
        y_remembered = y_pooled.iloc[bootstrap_indices].reset_index(drop=True)
        if C_pooled is not None:
            C_remembered = C_pooled.iloc[bootstrap_indices].reset_index(drop=True)
        else:
            C_remembered = None

        estimator_remembered = self.estimator.clone()
        estimator_remembered.fit(X=X_remembered, y=y_remembered, C=C_remembered)

        self.estimator_remembered_ = estimator_remembered
        self.estimator_pooled_ = estimator_pooled
        self._remembered_X = X_remembered
        self._remembered_y = y_remembered
        self._remembered_C = C_remembered
        self._n_remembered = n_remember
        self._n_pooled = n_pooled
        self._n_new = n_new

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

        params = [
            {"estimator": regressor},
            {"estimator": regressor, "n_remember": 50, "random_state": 42},
        ]

        if _check_estimator_deps(CoxPH, severity="none"):
            coxph = CoxPH()
            params.append({"estimator": coxph})
        else:
            ridge = Ridge()
            params.append({"estimator": ResidualDouble(ridge)})

        return params
