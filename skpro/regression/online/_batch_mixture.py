"""Meta-strategy for online learning: fit separate regressors on batches."""

__all__ = ["OnlineBatchMixture"]

import pandas as pd

from skpro.distributions.mixture import Mixture
from skpro.regression.base import _DelegatedProbaRegressor


class OnlineBatchMixture(_DelegatedProbaRegressor):
    """Online regression strategy by fitting separate regressors on batches.

    Fits separate copies of the regressor on each batch of data.
    Returns a Mixture distribution with weights proportional to the number
    of samples in each batch.

    Batches under a certain size can be ignored or accumulated until a
    minimum size is reached.

    Parameters
    ----------
    estimator : skpro regressor, descendant of BaseProbaRegressor
        regressor to be fitted on each batch, blueprint
    min_batch_size : int, default=1
        Minimum batch size to fit a regressor. Batches smaller than this
        are either ignored or accumulated (depending on ignore_small_batches).
    ignore_small_batches : bool, default=True
        If True, batches smaller than min_batch_size are ignored.
        If False, batches are accumulated until min_batch_size is reached.

    Attributes
    ----------
    estimators_ : list of skpro regressors
        list of fitted regressors, one per batch
    batch_sizes_ : list of int
        list of sample counts per batch, corresponding to estimators_
    """

    _tags = {"capability:update": True}

    def __init__(self, estimator, min_batch_size=1, ignore_small_batches=True):
        self.estimator = estimator
        self.min_batch_size = min_batch_size
        self.ignore_small_batches = ignore_small_batches

        super().__init__()

        tags_to_clone = [
            "capability:missing",
            "capability:survival",
        ]
        self.clone_tags(estimator, tags_to_clone)

    def _fit(self, X, y, C=None):
        """Fit regressor to training data.

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

        estimator = self.estimator.clone()
        estimator.fit(X=X, y=y, C=C)
        self.estimators_ = [estimator]
        self.batch_sizes_ = [n_samples]

        self._pending_X = None
        self._pending_y = None
        self._pending_C = None

        return self

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
        n_samples = len(X)

        if not self.ignore_small_batches:
            if self._pending_X is not None:
                X = self._update_data(self._pending_X, X)
                y = self._update_data(self._pending_y, y)
                C = self._update_data(self._pending_C, C)
                n_samples = len(X)
            else:
                self._pending_X = X
                self._pending_y = y
                self._pending_C = C

        if n_samples < self.min_batch_size:
            if self.ignore_small_batches:
                return self
            else:
                self._pending_X = X
                self._pending_y = y
                self._pending_C = C
                return self

        estimator = self.estimator.clone()
        estimator.fit(X=X, y=y, C=C)
        self.estimators_.append(estimator)
        self.batch_sizes_.append(n_samples)

        self._pending_X = None
        self._pending_y = None
        self._pending_C = None

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

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        Returns a Mixture distribution combining predictions from all batch regressors,
        with weights proportional to the number of samples in each batch.

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
        if len(self.estimators_) == 0:
            raise ValueError("No estimators fitted. At least one batch must be fitted.")

        distributions = []
        for i, est in enumerate(self.estimators_):
            pred_dist = est.predict_proba(X)
            distributions.append((f"batch_{i}", pred_dist))

        weights = self.batch_sizes_

        return Mixture(distributions=distributions, weights=weights)

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
            {
                "estimator": regressor,
                "min_batch_size": 5,
                "ignore_small_batches": False,
            },
        ]

        if _check_estimator_deps(CoxPH, severity="none"):
            coxph = CoxPH()
            params.append({"estimator": coxph})
        else:
            ridge = Ridge()
            params.append({"estimator": ResidualDouble(ridge)})

        return params
