"""Naive probabilistic regressor using distribution fitter."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["patelchaitany"]
__all__ = ["NaiveProbaRegressor"]

import numpy as np

from skpro.regression.base import BaseProbaRegressor


class NaiveProbaRegressor(BaseProbaRegressor):
    """Naive probabilistic regressor using a distribution fitter.

    Pools all training target values and fits a single distribution
    using the provided distribution fitter. At prediction time, the fitted
    distribution is broadcast to match the number of prediction instances.

    This regressor ignores the input features ``X`` entirely and always
    returns the same distribution for every prediction instance. It serves
    as a simple baseline to compare against more complex regressors.

    Parameters
    ----------
    distfitter : skpro distribution fitter instance, optional
        A distribution fitter from ``skpro.distfitter``.
        Default is ``NormalFitter(method="unbiased")``.

    Attributes
    ----------
    distfitter_ : skpro distribution fitter
        Clone of ``distfitter``, fitted to the training target values.
    distribution_ : skpro BaseDistribution
        Scalar distribution returned by the fitted ``distfitter_``.

    Examples
    --------
    >>> from skpro.regression.naive import NaiveProbaRegressor
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split

    Default usage with NormalFitter:

    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> reg = NaiveProbaRegressor()
    >>> reg.fit(X_train, y_train)
    NaiveProbaRegressor(...)
    >>> y_pred_proba = reg.predict_proba(X_test)

    Using a custom distribution fitter:

    >>> from skpro.distfitter import MOMFitter
    >>> from skpro.distributions.laplace import Laplace
    >>> fitter = MOMFitter(dist=Laplace, mean_name="mu", std_name="scale")
    >>> reg = NaiveProbaRegressor(distfitter=fitter)
    >>> reg.fit(X_train, y_train)
    NaiveProbaRegressor(...)
    """

    _tags = {
        "authors": ["patelchaitany"],
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, distfitter=None):
        self.distfitter = distfitter
        super().__init__()

    def _fit(self, X, y):
        """Fit regressor by fitting a distribution to the target values.

        Parameters
        ----------
        X : pandas DataFrame
            Feature instances (ignored).
        y : pandas DataFrame
            Target values to fit the distribution to.

        Returns
        -------
        self : reference to self
        """
        from skpro.distfitter import NormalFitter

        self._y_columns = y.columns

        if self.distfitter is None:
            distfitter = NormalFitter(method="unbiased")
        else:
            distfitter = self.distfitter.clone()

        distfitter.fit(y)
        self.distfitter_ = distfitter
        self.distribution_ = distfitter.proba()

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        Parameters
        ----------
        X : pandas DataFrame
            Feature instances to predict for.

        Returns
        -------
        y_pred : pandas DataFrame
            Point predictions (mean of the fitted distribution).
        """
        dist = self._predict_proba(X)
        return dist.mean()

    def _predict_proba(self, X):
        """Predict distribution for data from features.

        Broadcasts the scalar fitted distribution to match the number
        of instances in ``X``.

        Parameters
        ----------
        X : pandas DataFrame
            Feature instances to predict for.

        Returns
        -------
        y_pred_proba : skpro BaseDistribution
            Predicted distribution, same number of rows as ``X``.
        """
        X_ind = X.index
        X_n_rows = X.shape[0]

        dist = self.distribution_
        params = dist.get_params()

        broadcast_params = {}
        for key, val in params.items():
            if key in ("index", "columns"):
                continue
            if isinstance(val, (int, float, np.integer, np.floating)):
                broadcast_params[key] = np.full((X_n_rows, 1), val)
            else:
                broadcast_params[key] = val

        broadcast_params["index"] = X_ind
        broadcast_params["columns"] = self._y_columns

        pred_dist = dist.__class__(**broadcast_params)
        return pred_dist

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
        from skpro.distfitter import MOMFitter, NormalFitter
        from skpro.distributions.laplace import Laplace

        params1 = {}
        params2 = {"distfitter": NormalFitter(method="MLE")}
        params3 = {
            "distfitter": MOMFitter(dist=Laplace, mean_name="mu", std_name="scale")
        }

        return [params1, params2, params3]
