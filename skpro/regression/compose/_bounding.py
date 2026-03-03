"""Bounding wrapper for probabilistic regressors."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["arnavk23"]
__all__ = ["BoundingRegressor"]

from sklearn import clone

from skpro.regression.base import BaseProbaRegressor


class BoundingRegressor(BaseProbaRegressor):
    """Bounding wrapper for probabilistic regressors.

    Wraps an ``skpro`` probabilistic regressor and bounds/clips its predictions
    to be within specified lower and/or upper bounds.

    The bounding is applied at the point prediction level for ``predict``, and
    at the distributional level for ``predict_proba`` through truncation.

    Parameters
    ----------
    estimator : skpro regressor
        Probabilistic regressor to wrap and bound.
    lower : float or None, default=None
        Lower bound for predictions. If None, no lower bound is applied.
    upper : float or None, default=None
        Upper bound for predictions. If None, no upper bound is applied.
    method : str, default="truncate"
        Method to apply bounding to distributions:

        * "truncate": uses truncated distributions (recommended)
        * "clip_mean": clips the mean only, keeps original variance
        * "delta": replaces with delta distributions at the bounds (conservative)

    Attributes
    ----------
    estimator_ : skpro regressor
        Fitted clone of the wrapped estimator.

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from skpro.regression.parametric import ParametricRegressor
    >>> from skpro.regression.compose import BoundingRegressor
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> # Create a probabilistic regressor
    >>> reg = ParametricRegressor(LinearRegression(), distr="Normal")
    >>>
    >>> # Wrap it with bounds (e.g., for positive predictions)
    >>> bounded_reg = BoundingRegressor(reg, lower=0.0)
    >>> bounded_reg.fit(X_train, y_train)
    BoundingRegressor(...)
    >>> y_pred = bounded_reg.predict(X_test)
    >>> y_pred_proba = bounded_reg.predict_proba(X_test)
    """

    _tags = {
        "authors": ["arnavk23"],
        "capability:missing": True,
    }

    def __init__(self, estimator, lower=None, upper=None, method="truncate"):
        self.estimator = estimator
        self.lower = lower
        self.upper = upper
        self.method = method
        super().__init__()

        # Validate bounds
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError(
                f"lower bound ({lower}) must be less than upper bound ({upper})"
            )

    def _fit(self, X, y):
        """Fit the bounding regressor.

        Parameters
        ----------
        X : pandas DataFrame
            Feature instances to fit regressor to.
        y : pandas DataFrame
            Labels to fit regressor to (must be same length as X).

        Returns
        -------
        self : reference to self
        """
        # Fit the wrapped estimator
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y)

        # Store column names
        self._y_cols = y.columns

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame
            Data to predict labels for.

        Returns
        -------
        y : pandas DataFrame
            Predictions of target values for X, clipped to bounds.
        """
        y_pred = self.estimator_.predict(X)

        # Clip predictions to bounds
        if self.lower is not None:
            y_pred = y_pred.clip(lower=self.lower)
        if self.upper is not None:
            y_pred = y_pred.clip(upper=self.upper)

        return y_pred

    def _predict_proba(self, X):
        """Predict distribution over labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame
            Data to predict labels for.

        Returns
        -------
        y_proba : skpro BaseDistribution
            Predictive distribution for X, bounded/truncated.
        """
        # Get unbounded distribution
        y_dist = self.estimator_.predict_proba(X)

        if self.method == "truncate":
            # Use truncation to bound the distribution
            from skpro.distributions.truncated import TruncatedDistribution

            # Apply truncation if bounds are specified
            if self.lower is not None or self.upper is not None:
                y_dist_bounded = TruncatedDistribution(
                    y_dist,
                    lower=self.lower,
                    upper=self.upper,
                )
                return y_dist_bounded
            else:
                return y_dist

        elif self.method == "clip_mean":
            # Get the mean and clip it, but keep the distribution type
            mean = y_dist.mean()

            # Clip the mean
            if self.lower is not None:
                mean = mean.clip(lower=self.lower)
            if self.upper is not None:
                mean = mean.clip(upper=self.upper)

            # Try to create a new distribution with the clipped mean
            # This is a simplified approach - in practice might need
            # more sophisticated handling per distribution type
            try:
                from skpro.distributions.normal import Normal

                std = y_dist.std()
                return Normal(mu=mean, sigma=std, index=X.index, columns=self._y_cols)
            except Exception:
                # Fallback: return truncated version
                from skpro.distributions.truncated import TruncatedDistribution

                return TruncatedDistribution(
                    y_dist, lower=self.lower, upper=self.upper
                )

        elif self.method == "delta":
            # Replace out-of-bounds predictions with delta distributions
            from skpro.distributions.delta import Delta

            mean = y_dist.mean()

            # Clip the mean
            if self.lower is not None:
                mean = mean.clip(lower=self.lower)
            if self.upper is not None:
                mean = mean.clip(upper=self.upper)

            return Delta(c=mean, index=X.index, columns=self._y_cols)

        else:
            raise ValueError(
                f"Unknown bounding method: {self.method}. "
                "Must be one of ['truncate', 'clip_mean', 'delta']"
            )

    def _predict_interval(self, X, coverage=0.9):
        """Predict interval for data from features.

        Parameters
        ----------
        X : pandas DataFrame
            Data to predict intervals for.
        coverage : float or list of float, default=0.9
            Nominal coverage of the interval.

        Returns
        -------
        pred_int : pandas DataFrame
            Interval predictions, clipped to bounds.
        """
        pred_int = self.estimator_.predict_interval(X, coverage=coverage)

        # Clip interval bounds
        if self.lower is not None:
            # Clip lower bounds of intervals
            lower_cols = [col for col in pred_int.columns if "lower" in str(col)]
            pred_int[lower_cols] = pred_int[lower_cols].clip(lower=self.lower)

            # Also clip upper bounds
            upper_cols = [col for col in pred_int.columns if "upper" in str(col)]
            pred_int[upper_cols] = pred_int[upper_cols].clip(lower=self.lower)

        if self.upper is not None:
            # Clip upper bounds of intervals
            upper_cols = [col for col in pred_int.columns if "upper" in str(col)]
            pred_int[upper_cols] = pred_int[upper_cols].clip(upper=self.upper)

            # Also clip lower bounds
            lower_cols = [col for col in pred_int.columns if "lower" in str(col)]
            pred_int[lower_cols] = pred_int[lower_cols].clip(upper=self.upper)

        return pred_int

    def _predict_quantiles(self, X, alpha):
        """Predict quantiles for data from features.

        Parameters
        ----------
        X : pandas DataFrame
            Data to predict quantiles for.
        alpha : list of float
            Quantile levels to predict.

        Returns
        -------
        pred_q : pandas DataFrame
            Quantile predictions, clipped to bounds.
        """
        pred_q = self.estimator_.predict_quantiles(X, alpha=alpha)

        # Clip quantile predictions to bounds
        if self.lower is not None:
            pred_q = pred_q.clip(lower=self.lower)
        if self.upper is not None:
            pred_q = pred_q.clip(upper=self.upper)

        return pred_q

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        from sklearn.linear_model import LinearRegression

        from skpro.regression.parametric import ParametricRegressor

        base_est = ParametricRegressor(LinearRegression(), distr="Normal")

        params1 = {"estimator": base_est, "lower": 0.0}
        params2 = {"estimator": base_est, "lower": 0.0, "upper": 100.0}
        params3 = {
            "estimator": base_est,
            "lower": 0.0,
            "method": "clip_mean",
        }

        return [params1, params2, params3]
