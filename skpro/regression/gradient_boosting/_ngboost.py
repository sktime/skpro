"""Adapters to ngboost regressors with probabilistic components."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["ShreeshaM07"]

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.learners import default_tree_learner
from ngboost.manifold import manifold
from ngboost.scores import LogScore
from sklearn.utils import check_random_state

from skpro.regression.base import BaseProbaRegressor


class NGBoostRegressor(BaseProbaRegressor):
    """
    Constructor for all NGBoost models.

    This class implements the methods that are common to all NGBoost models.
    Unless you are implementing a new kind of regression (e.g. interval-censored, etc.),
    you should probably use one of NGBRegressor, NGBClassifier, or NGBSurvival.

    Parameters
    ----------
        Dist              : assumed distributional form of Y|X=x.
                            A distribution from ngboost.distns, e.g. Normal
        Score             : rule to compare probabilistic
                            predictions P̂ to the observed data y.
                            A score from ngboost.scores, e.g. LogScore
        Base              : base learner to use in the boosting algorithm.
                            Any instantiated sklearn regressor,
                            e.g. DecisionTreeRegressor()
        natural_gradient  : logical flag indicating whether the natural
                            gradient should be used
        n_estimators      : the number of boosting iterations to fit
        learning_rate     : the learning rate
        minibatch_frac    : the percent subsample of rows to
                            use in each boosting iteration
        verbose           : flag indicating whether output
                            should be printed during fitting
        verbose_eval      : increment (in boosting iterations) at
                            which output should be printed
        tol               : numerical tolerance to be used in optimization
        random_state      : seed for reproducibility.
        validation_fraction: Proportion of training data to
                             set aside as validation data for early stopping.
        early_stopping_rounds: The number of consecutive
            boosting iterations during which the
            loss has to increase before the algorithm stops early.
            Set to None to disable early stopping and validation
            None enables running over the full data set.


    Returns
    -------
        An NGBRegressor object that can be fit.
    """

    def __init__(
        self,
        Dist=Normal,
        Score=LogScore,
        Base=default_tree_learner,
        natural_gradient=True,
        n_estimators=500,
        learning_rate=0.01,
        minibatch_frac=1.0,
        col_sample=1.0,
        verbose=True,
        verbose_eval=100,
        tol=1e-4,
        random_state=None,
        validation_fraction=0.1,
        early_stopping_rounds=None,
    ):
        self.Dist = Dist
        self.Score = Score
        self.Base = Base
        self.Manifold = manifold(Score, Dist)
        self.natural_gradient = natural_gradient
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.verbose = verbose
        self.verbose_eval = verbose_eval
        self.init_params = None
        self.n_features = None
        self.base_models = []
        self.scalings = []
        self.col_idxs = []
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.best_val_loss_itr = None
        self.validation_fraction = validation_fraction
        self.early_stopping_rounds = early_stopping_rounds

        super().__init__()

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
        self.ngb = NGBRegressor(
            Dist=self.Dist,
            Score=self.Score,
            Base=default_tree_learner,
            natural_gradient=True,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            minibatch_frac=self.minibatch_frac,
            col_sample=self.col_sample,
            verbose=self.verbose,
            verbose_eval=self.verbose_eval,
            tol=self.tol,
            random_state=self.random_state,
            validation_fraction=self.validation_fraction,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.ngb.fit(X, y)
        self._is_fitted = True
        return self

    def _predict(self, X):
        return self.ngb.predict(X)

    def _pred_dist(self, X):
        return self.ngb.pred_dist(X)

    # def _ngb_dist_to_skpro(self, ngb_dist):
    #     """
    #     Convert NGBoost distribution object to skpro BaseDistribution object.

    #     Parameters:
    #     ngb_dist (ngboost.distns.Distribution): NGBoost distribution object.

    #     Returns:
    #     skpro_dist (skpro.distributions.BaseDistribution):
    #     Converted skpro distribution object.
    #     """
    #     from skpro.distributions.normal import Normal
    #     if isinstance(ngb_dist, Normal):
    #         mu = ngb_dist.params['loc']
    #         sigma = np.sqrt(ngb_dist.params['scale'])
    #         skpro_dist = Normal(mu=mu, sigma=sigma)
    #     else:
    #         raise NotImplementedError("Conversion for
    #         this distribution is not implemented.")

    #     return skpro_dist

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
        X = self._check_X(X)
        # default behaviour is implemented if one of the following three is implemented
        implements_interval = self._has_implementation_of("_predict_interval")
        implements_quantiles = self._has_implementation_of("_predict_quantiles")
        implements_var = self._has_implementation_of("_predict_var")
        can_do_proba = implements_interval or implements_quantiles or implements_var

        if not can_do_proba:
            raise NotImplementedError

        # defaulting logic is as follows:
        # var direct deputies are proba, then interval
        # proba direct deputy is var (via Normal dist)
        # quantiles direct deputies are interval, then proba
        # interval direct deputy is quantiles
        #
        # so, conditions for defaulting for proba is:
        # default to var if any of the other three are implemented

        # we use predict_var to get scale, and predict to get location
        # pred_dist = self._ngb_dist_to_skpro(self._pred_dist(X))

        # Convert NGBoost distribution to skpro BaseDistribution
        pred_mean = self.ngb.predict(X=X)
        pred_std = np.sqrt(self._pred_dist(X).params["scale"])
        pred_std = self._check_y(y=pred_std)
        # returns a tuple so taking only first index of the tuple
        pred_std = pred_std[0]

        pred_mean = self._check_y(y=pred_mean)
        # returns a tuple so taking only first index of the tuple
        pred_mean = pred_mean[0]
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
        implements_var = self._has_implementation_of("_predict_var")
        can_do_proba = implements_quantiles or implements_proba or implements_var

        if not can_do_proba:
            raise NotImplementedError

        # defaulting logic is as follows:
        # var direct deputies are proba, then interval
        # proba direct deputy is var (via Normal dist)
        # quantiles direct deputies are interval, then proba
        # interval direct deputy is quantiles
        #
        # so, conditions for defaulting for interval are:
        # default to quantiles if any of the other three methods are implemented

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
        implements_var = self._has_implementation_of("_predict_var")
        can_do_proba = implements_interval or implements_proba or implements_var

        if not can_do_proba:
            raise NotImplementedError

        # defaulting logic is as follows:
        # var direct deputies are proba, then interval
        # proba direct deputy is var (via Normal dist)
        # quantiles direct deputies are interval, then proba
        # interval direct deputy is quantiles
        #
        # so, conditions for defaulting for quantiles are:
        # 1. default to interval if interval implemented
        # 2. default to proba if proba or var are implemented

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

        elif implements_proba or implements_var:
            pred_proba = self._predict_proba(X=X)
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

        # defaulting logic is as follows:
        # var direct deputies are proba, then interval
        # proba direct deputy is var (via Normal dist)
        # quantiles direct deputies are interval, then proba
        # interval direct deputy is quantiles
        #
        # so, conditions for defaulting for var are:
        # 1. default to proba if proba implemented
        # 2. default to interval if interval or quantiles are implemented

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


# #Load Boston housing dataset
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# Y = raw_df.values[1::2, 2]

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
# ngb = NGBoostRegressor()._fit(X_train, Y_train)
# Y_preds = ngb._predict(X_test)

# Y_dists = ngb._pred_dist(X_test)

# print(Y_dists)
# Y_pred_proba = ngb._predict_proba(X_test)
# print(Y_pred_proba)

# # test Mean Squared Error
# test_MSE = mean_squared_error(Y_preds, Y_test)
# print('Test MSE', test_MSE)

# # test Negative Log Likelihood
# test_NLL = -Y_dists.logpdf(Y_test).mean()
# print('Test NLL', test_NLL)
