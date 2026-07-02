"""Dummy time series regressor."""

__author__ = ["julian-fong"]
__all__ = ["DummyProbaRegressor"]

import warnings

import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.optimize import brentq

from skpro.distributions.empirical import Empirical
from skpro.distributions.normal import Normal
from skpro.regression.base import BaseProbaRegressor


class DummyProbaRegressor(BaseProbaRegressor):
    """DummyProbaRegressor makes predictions that ignore the input features.

    This regressor serves as a simple baseline to compare against other more
    complex regressors.
    The specific behavior of the baseline is selected with the ``strategy``
    parameter.

    All strategies make predictions that ignore the input feature values passed
    as the ``X`` argument to ``fit`` and ``predict``. The predictions, however,
    typically depend on values observed in the ``y`` parameter passed to ``fit``.

    Parameters
    ----------
    strategy : one of ["empirical", "normal", "kernel"] default="empirical"
        Strategy to use to generate predictions.

        * "empirical": always predicts the empirical unweighted distribution
            of the training labels
        * "normal": always predicts a normal distribution, with mean and variance
            equal to the mean and variance of the training labels
        * "kernel": always predicts a kernel density estimate based on the
            training labels, using Gaussian kernels

    bandwidth : float or str, optional, default="scott"
        Bandwidth parameter for kernel density estimation, only used when
        ``strategy="kernel"``. Can be a float for fixed bandwidth, or one of:
        "scott", "silverman", "isj" for automatic bandwidth selection.
        See ``scipy.stats.gaussian_kde`` for details.

    Notes (Kernel Strategy)
    ----------------------
    * Improved Sheather-Jones is well-suited for multimodal data
      but requires sufficient samples (n >= 50). With very small datasets,
      computation may fail and fall back to Silverman's rule with a warning.
    * KDE bandwidth selection is heuristic and does not guarantee
      any specific coverage level. ISJ targets asymptotic optimality.
    * With samples having very small variance or degenerate data,
      KDE bandwidth computation may be unstable.

    n_kde_samples : int, default=1000
        Number of samples to draw from the kernel density estimate for
        empirical distribution approximation, only used when ``strategy="kernel"``.

    Attributes
    ----------
    distribution_ : skpro.distribution
        Normal distribution or Empirical distribution, depending on chosen strategy.
        Scalar version of the distribution that is returned by ``predict_proba``.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["julian-fong"],
        "maintainers": ["julian-fong", "arnavk23"],
        # estimator tags
        # --------------
        "capability:multioutput": False,
        "capability:missing": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, strategy="empirical", bandwidth="scott", n_kde_samples=1000):
        self.strategy = strategy
        self.bandwidth = bandwidth
        self.n_kde_samples = n_kde_samples
        super().__init__()

    def _fit(self, X, y):
        """Fit the dummy regressor.

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
        self._y = y
        self._y_columns = y.columns
        self._mu = np.mean(y.values)
        self._sigma = np.std(y.values)

        if self.strategy == "empirical":
            self.distribution_ = Empirical(y)
        elif self.strategy == "normal":
            self.distribution_ = Normal(self._mu, self._sigma)
        elif self.strategy == "kernel":
            from scipy.stats import gaussian_kde

            # Fit KDE to the training data
            y_values = y.values
            bw_method = _resolve_kde_bw_method(y_values, self.bandwidth)
            self._kde = gaussian_kde(y_values.flatten(), bw_method=bw_method)

            # Sample from KDE to create empirical distribution
            np.random.seed(42)  # For reproducibility
            kde_samples = self._kde.resample(self.n_kde_samples).T
            kde_df = pd.DataFrame(kde_samples, columns=self._y_columns)
            self.distribution_ = Empirical(kde_df)
        else:
            raise ValueError(
                f"Unknown strategy: {self.strategy}. "
                f"Must be one of ['empirical', 'normal', 'kernel']"
            )

        return self

    def _predict(self, X):
        """Predict labels for data from features.

        Parameters
        ----------
        X : sktime-format pandas dataframe or array-like, shape (n, d)

        Returns
        -------
        y : pandas DataFrame
            predictions of target values for X
        """
        X_ind = X.index
        X_n_rows = X.shape[0]
        y_pred = pd.DataFrame(
            np.ones(X_n_rows) * self._mu, index=X_ind, columns=self._y_columns
        )
        return y_pred

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
        X_ind = X.index
        X_n_rows = X.shape[0]
        y_pred = pd.DataFrame(
            np.ones(X_n_rows) * self._sigma, index=X_ind, columns=self._y_columns
        )
        return y_pred

    def _predict_proba(self, X):
        """Broadcast skpro distribution from fit onto labels from X.

        Parameters
        ----------
        X : sktime-format pandas dataframe or array-like, shape (n, d)

        Returns
        -------
        y : skpro.distribution, same length as `X`
            labels predicted for `X`
        """
        X_ind = X.index
        X_n_rows = X.shape[0]

        if self.strategy == "normal":
            # broadcast the mu and sigma from fit to the length of X
            mu = np.reshape((np.ones(X_n_rows) * self._mu), (-1, 1))
            sigma = np.reshape((np.ones(X_n_rows) * self._sigma), (-1, 1))
            pred_dist = Normal(mu=mu, sigma=sigma, index=X_ind, columns=self._y_columns)
            return pred_dist

        elif self.strategy == "empirical":
            empr_df = pd.concat([self._y] * X_n_rows, keys=X_ind).swaplevel()
            pred_dist = Empirical(empr_df, index=X_ind, columns=self._y_columns)
            return pred_dist

        elif self.strategy == "kernel":
            # Sample from KDE for each prediction instance
            kde_samples_list = []
            for _ in range(X_n_rows):
                samples = self._kde.resample(self.n_kde_samples).T
                kde_samples_list.append(samples)

            # Stack samples and create empirical distribution
            all_samples = np.vstack(kde_samples_list)
            sample_index = pd.MultiIndex.from_product(
                [range(self.n_kde_samples), X_ind], names=["sample", "instance"]
            ).swaplevel()
            kde_df = pd.DataFrame(
                all_samples, index=sample_index, columns=self._y_columns
            )
            pred_dist = Empirical(kde_df, index=X_ind, columns=self._y_columns)
            return pred_dist

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
        params1 = {}
        params2 = {"strategy": "normal"}
        params3 = {"strategy": "kernel", "n_kde_samples": 100}

        return [params1, params2, params3]


def _resolve_kde_bw_method(y_values, bandwidth):
    """Resolve bandwidth option for scipy.stats.gaussian_kde."""
    if isinstance(bandwidth, str):
        method = bandwidth.lower()
        if method in {"scott", "silverman"}:
            return method
        if method == "isj":
            arr = np.asarray(y_values)
            if arr.ndim > 1 and arr.shape[1] > 1:
                warnings.warn(
                    "ISJ bandwidth is currently only supported for 1D targets. "
                    "Falling back to silverman for multi-column targets.",
                    UserWarning,
                    stacklevel=2,
                )
                return "silverman"
            y_1d = arr.reshape(-1)
            return _isj_bw_factor_1d(y_1d)
        raise ValueError(
            f"Unknown bandwidth: {bandwidth}. "
            "Must be a positive float or one of ['scott', 'silverman', 'isj']."
        )

    h = float(bandwidth)
    if h <= 0:
        raise ValueError("bandwidth must be positive")
    return h


def _isj_bw_factor_1d(y):
    """Return gaussian_kde bandwidth factor from ISJ absolute bandwidth in 1D.

    Notes
    -----
    ISJ bandwidth requires at least 50 samples for stable numerical behavior.
    For very small datasets, may fall back to Silverman's rule with a warning.
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    sigma = np.std(y, ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        return "silverman"

    if len(y) < 50:
        warnings.warn(
            f"ISJ bandwidth with only {len(y)} samples may be numerically unstable. "
            "For robust results, use at least 50 samples. "
            "Consider using Silverman's rule instead.",
            UserWarning,
            stacklevel=2,
        )

    h = _bw_isj_1d(y)

    if not np.isfinite(h) or h <= 0:
        warnings.warn(
            "ISJ produced a non-positive bandwidth; falling back to silverman.",
            UserWarning,
            stacklevel=2,
        )
        return "silverman"

    return float(h / sigma)


def _bw_isj_1d(y):
    """Compute 1D ISJ bandwidth, adapted from KDEpy's bw_selection implementation."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size < 2:
        raise ValueError("ISJ bandwidth requires at least 2 points")

    n = 2**10
    xmesh = _autogrid_1d(y, num_points=n)
    R = np.max(y) - np.min(y)
    if not np.isfinite(R) or R <= 0:
        raise ValueError("ISJ bandwidth is undefined for degenerate data")

    N = len(np.unique(y))
    initial_data = _linear_binning_1d(y, xmesh)
    if not np.isclose(initial_data.sum(), 1.0):
        initial_data = initial_data / initial_data.sum()

    a = fftpack.dct(initial_data)
    I_sq = np.power(np.arange(1, n, dtype=float), 2)
    a2 = a[1:] ** 2

    t_star = _root_isj(_fixed_point_isj, N, args=(N, I_sq, a2))
    return float(np.sqrt(t_star) * R)


def _autogrid_1d(y, num_points=1024):
    """Create equidistant grid for 1D ISJ bandwidth computation."""
    y = np.asarray(y, dtype=float)
    ymin = np.min(y)
    ymax = np.max(y)
    R = ymax - ymin
    sigma = np.std(y, ddof=1)

    if not np.isfinite(R) or R <= 0:
        span = sigma if np.isfinite(sigma) and sigma > 0 else 1.0
    else:
        span = max(
            0.5 * R, 6.0 * sigma if np.isfinite(sigma) and sigma > 0 else 0.5 * R
        )

    lo = ymin - span
    hi = ymax + span
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = ymin - 1.0, ymax + 1.0
    if hi <= lo:
        lo, hi = -1.0, 1.0
    return np.linspace(lo, hi, int(num_points))


def _linear_binning_1d(y, xmesh):
    """Linear binning of samples on equidistant 1D mesh."""
    y = np.asarray(y, dtype=float).reshape(-1)
    xmesh = np.asarray(xmesh, dtype=float).reshape(-1)

    n = xmesh.size
    if n < 2:
        raise ValueError("xmesh must have at least 2 points")

    dx = xmesh[1] - xmesh[0]
    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("xmesh must be strictly increasing and equidistant")

    out = np.zeros(n, dtype=float)
    scaled = (y - xmesh[0]) / dx
    left = np.floor(scaled).astype(int)
    frac = scaled - left
    w = 1.0 / y.size

    left_ok = (left >= 0) & (left < n)
    np.add.at(out, left[left_ok], (1.0 - frac[left_ok]) * w)

    right = left + 1
    right_ok = (right >= 0) & (right < n)
    np.add.at(out, right[right_ok], frac[right_ok] * w)

    s = out.sum()
    if s <= 0:
        raise ValueError("linear binning produced empty bins")
    return out / s


def _fixed_point_isj(t, N, I_sq, a2):
    """Fixed-point function in Botev's ISJ algorithm."""
    I_sq = np.asarray(I_sq, dtype=float)
    a2 = np.asarray(a2, dtype=float)

    ell = 7
    f = (
        0.5
        * np.pi ** (2 * ell)
        * np.sum(np.power(I_sq, ell) * a2 * np.exp(-I_sq * np.pi**2 * t))
    )
    if f <= 0 or not np.isfinite(f):
        return -1.0

    for s in reversed(range(2, ell)):
        odd_numbers_prod = np.prod(np.arange(1, 2 * s + 1, 2, dtype=float))
        K0 = odd_numbers_prod / np.sqrt(2 * np.pi)
        const = (1 + (1 / 2) ** (s + 0.5)) / 3
        time = np.power((2 * const * K0 / (N * f)), (2.0 / (3.0 + 2.0 * s)))
        f = (
            0.5
            * np.pi ** (2 * s)
            * np.sum(np.power(I_sq, s) * a2 * np.exp(-I_sq * np.pi**2 * time))
        )

    t_opt = np.power(2 * N * np.sqrt(np.pi) * f, -2.0 / 5)
    return t - t_opt


def _root_isj(function, N, args):
    """Root finder for ISJ fixed-point equation, robust to difficult cases."""
    N = max(min(1050, N), 50)
    tol = 1e-11 + 0.01 * (N - 50) / 1000
    lower = 0.0
    f_lower = function(lower, *args)

    for _ in range(20):
        upper = tol
        f_upper = function(upper, *args)
        if np.isfinite(f_lower) and np.isfinite(f_upper) and f_lower * f_upper <= 0:
            return brentq(function, lower, upper, args=args, xtol=upper)
        tol *= 2

    return np.nan
