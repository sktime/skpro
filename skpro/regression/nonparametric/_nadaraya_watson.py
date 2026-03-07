"""Nadaraya-Watson conditional density estimator."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["patelchaitany93"]
__all__ = ["NadarayaWatsonCDE"]

import numpy as np
import pandas as pd

from skpro.distributions.empirical import Empirical
from skpro.regression.base import BaseProbaRegressor


class NadarayaWatsonCDE(BaseProbaRegressor):
    r"""Nadaraya-Watson conditional density estimator.

    Non-parametric probabilistic regressor based on kernel-weighted
    averaging of training targets. For each test point, the predicted
    conditional distribution is a weighted empirical distribution over
    the training targets, where the weights are determined by a kernel
    function applied to the distance between test and training features.

    For a test point :math:`x`, the conditional density is estimated as:

    .. math::

        \hat{f}(y \mid x)
        = \sum_{i=1}^{n} w_i(x) \, \delta(y - y_i),
        \quad
        w_i(x) = \frac{K_h(\|x - X_i\|)}{\sum_{j=1}^{n} K_h(\|x - X_j\|)}

    where :math:`K_h` is a kernel function with bandwidth :math:`h`,
    :math:`(X_i, y_i)` are the training samples, and :math:`\delta` is
    the Dirac delta.

    Parameters
    ----------
    bandwidth : float or {"scott", "silverman"}, default="scott"
        Bandwidth for the kernel function.

        * If ``float``, used directly as the bandwidth parameter :math:`h`.
        * If ``"scott"``, bandwidth is computed by Scott's rule:
          :math:`h = n^{-1/(d+4)}` where :math:`n` is the number of
          training samples and :math:`d` is the number of features.
        * If ``"silverman"``, bandwidth is computed by Silverman's rule:
          :math:`h = (n \cdot (d+2) / 4)^{-1/(d+4)}`.

    kernel : {"gaussian", "epanechnikov", "uniform"}, default="gaussian"
        Kernel function to use for weighting.

        * ``"gaussian"``:
          :math:`K(u) = \exp(-u^2 / 2)`
        * ``"epanechnikov"``:
          :math:`K(u) = \max(0, 1 - u^2)`
        * ``"uniform"``:
          :math:`K(u) = \mathbf{1}(|u| \le 1)`

    normalize_features : bool, default=True
        Whether to standardize features (zero mean, unit variance) before
        computing distances. Recommended when features have different scales.

    Attributes
    ----------
    X_train_ : pd.DataFrame
        Training feature data stored after ``fit``.
    y_train_ : pd.DataFrame
        Training target data stored after ``fit``.
    bandwidth_ : float
        Effective bandwidth used after fitting (resolved from string rules).
    x_mean_ : np.ndarray
        Per-feature mean of training data (only if ``normalize_features=True``).
    x_std_ : np.ndarray
        Per-feature std of training data (only if ``normalize_features=True``).

    Examples
    --------
    >>> from skpro.regression.nonparametric import NadarayaWatsonCDE
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = load_diabetes(return_X_y=True, as_frame=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>>
    >>> reg = NadarayaWatsonCDE(bandwidth="scott")
    >>> reg.fit(X_train, y_train)
    NadarayaWatsonCDE(...)
    >>> y_pred_proba = reg.predict_proba(X_test)
    """

    _tags = {
        "authors": ["patelchaitany93"],
        "capability:multioutput": False,
        "capability:missing": False,
    }

    def __init__(self, bandwidth="scott", kernel="gaussian", normalize_features=True):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.normalize_features = normalize_features

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
        self._y_cols = y.columns

        X_arr = X.values.astype(float)

        if self.normalize_features:
            self.x_mean_ = X_arr.mean(axis=0)
            self.x_std_ = X_arr.std(axis=0)
            self.x_std_[self.x_std_ == 0] = 1.0
            X_arr = (X_arr - self.x_mean_) / self.x_std_

        self.X_train_ = X_arr
        self.y_train_ = y.copy()

        n, d = X_arr.shape
        bandwidth = self.bandwidth
        if bandwidth == "scott":
            self.bandwidth_ = n ** (-1.0 / (d + 4))
        elif bandwidth == "silverman":
            self.bandwidth_ = (n * (d + 2) / 4.0) ** (-1.0 / (d + 4))
        else:
            self.bandwidth_ = float(bandwidth)

        return self

    def _predict(self, X):
        """Predict expected value (conditional mean) for test data.

        Parameters
        ----------
        X : pandas DataFrame, must have same columns as X in `fit`
            data to predict labels for

        Returns
        -------
        y_pred : pandas DataFrame, same length as `X`
            predicted mean labels for `X`
        """
        weights = self._compute_weights(X)
        y_vals = self.y_train_.values

        y_pred = weights @ y_vals

        return pd.DataFrame(y_pred, index=X.index, columns=self._y_cols)

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
        weights = self._compute_weights(X)
        y_train = self.y_train_
        cols = self._y_cols
        n_train = len(y_train)
        n_test = len(X)

        spl_frames = []
        weight_series = []

        for i in range(n_test):
            w_i = weights[i]

            y_spl = y_train.copy()
            y_spl.index = pd.MultiIndex.from_arrays(
                [np.arange(n_train), np.full(n_train, X.index[i])],
                names=["sample", y_train.index.name],
            )

            w_ser = pd.Series(w_i, index=y_spl.index)

            spl_frames.append(y_spl)
            weight_series.append(w_ser)

        spl_df = pd.concat(spl_frames, axis=0)
        weights_all = pd.concat(weight_series, axis=0)

        y_proba = Empirical(
            spl=spl_df,
            weights=weights_all,
            index=X.index,
            columns=cols,
        )
        return y_proba

    def _compute_weights(self, X):
        """Compute kernel weights between test points and training points.

        Parameters
        ----------
        X : pandas DataFrame
            test feature data

        Returns
        -------
        weights : np.ndarray, shape (n_test, n_train)
            normalized kernel weights, each row sums to 1
        """
        X_arr = X.values.astype(float)

        if self.normalize_features:
            X_arr = (X_arr - self.x_mean_) / self.x_std_

        dists = self._pairwise_distances(X_arr, self.X_train_)

        scaled_dists = dists / self.bandwidth_

        kernel = self.kernel
        if kernel == "gaussian":
            k_vals = np.exp(-0.5 * scaled_dists**2)
        elif kernel == "epanechnikov":
            k_vals = np.maximum(0, 1 - scaled_dists**2)
        elif kernel == "uniform":
            k_vals = (scaled_dists <= 1).astype(float)
        else:
            raise ValueError(
                f"Unknown kernel: {kernel}. "
                "Must be one of 'gaussian', 'epanechnikov', 'uniform'."
            )

        row_sums = k_vals.sum(axis=1, keepdims=True)
        zero_mask = (row_sums == 0).flatten()
        row_sums[zero_mask] = 1.0
        weights = k_vals / row_sums
        weights[zero_mask] = 1.0 / k_vals.shape[1]
        return weights

    @staticmethod
    def _pairwise_distances(A, B):
        """Compute Euclidean distances between rows of A and rows of B.

        Parameters
        ----------
        A : np.ndarray, shape (n, d)
        B : np.ndarray, shape (m, d)

        Returns
        -------
        dists : np.ndarray, shape (n, m)
        """
        A_sq = np.sum(A**2, axis=1, keepdims=True)
        B_sq = np.sum(B**2, axis=1, keepdims=True)
        cross = A @ B.T
        dist_sq = A_sq - 2 * cross + B_sq.T
        dist_sq = np.maximum(dist_sq, 0)
        return np.sqrt(dist_sq)

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
        params1 = {"bandwidth": "scott", "kernel": "gaussian"}
        params2 = {"bandwidth": 0.5, "kernel": "epanechnikov"}
        params3 = {
            "bandwidth": "silverman",
            "kernel": "uniform",
            "normalize_features": False,
        }

        return [params1, params2, params3]
