# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Multivariate Normal probability distribution."""

__author__ = ["ParamThakkar123"]

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm

from skpro.distributions.adapters.scipy import _ScipyAdapter


class MultivariateNormal(_ScipyAdapter):
    """Multivariate Normal distribution.

    Parameters
    ----------
    mean : array-like
        Mean vector of the multivariate normal. Can be
        - 1D sequence (length d): a single d-dimensional mean (broadcasted)
        - 2D array-like (n_instances, d): one mean vector per row instance
    cov : array-like
        Covariance matrix (d x d) for a single distribution, or
        array-like of shape (n_instances, d, d) for one covariance per instance.
    index : pd.Index or None
        Row index for array-valued distributions. If omitted, inference is used.
    columns : pd.Index or None
        Column (variable) names / dimensional names. Length must equal d when provided.

    Returns
    -------
    MultivariateNormal instance

    Examples
    --------
    >>> from skpro.distributions import MultivariateNormal

    >>> distr = MultivariateNormal(mean=[0, 0], cov=[[1, 0], [0, 1]])
    """

    _tags = {
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
        "broadcast_params": ["mean", "cov"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "sample"],
        "capabilities:approx": ["pdfnorm", "energy"],
    }

    def __init__(self, mean, cov, index=None, columns=None):
        self.mean = mean
        self.cov = cov
        self.index = index
        self.columns = columns

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self):
        return multivariate_normal

    def _get_scipy_param(self):
        mean = self._bc_params["mean"]
        cov = self._bc_params["cov"]

        return [], {"mean": mean, "cov": cov}

    def _pdf(self, x):
        """Joint pdf evaluated row-wise; broadcast scalar joint pdf across columns."""
        mean = self._bc_params["mean"]
        cov = self._bc_params["cov"]
        x = np.asarray(x)

        # scalar distribution (0-d): pass through to scipy
        if self.ndim == 0:
            return multivariate_normal.pdf(x, mean=mean, cov=cov)

        n_row, n_col = self.shape
        out = np.zeros((n_row, n_col), dtype=float)

        # handle broadcasted param shapes
        mean_arr = np.asarray(mean)
        cov_arr = np.asarray(cov)
        for i in range(n_row):
            xi = x[i] if x.ndim == 2 else x
            mi = mean_arr[i] if mean_arr.ndim == 2 else mean_arr
            ci = cov_arr[i] if cov_arr.ndim == 3 else cov_arr
            p = multivariate_normal.pdf(xi, mean=mi, cov=ci)
            out[i, :] = p
        return out

    def _log_pdf(self, x):
        """Row-wise joint log-pdf, broadcast across columns to match framework shape."""
        mean = self._bc_params["mean"]
        cov = self._bc_params["cov"]
        x = np.asarray(x)

        if self.ndim == 0:
            return multivariate_normal.logpdf(x, mean=mean, cov=cov)

        n_row, n_col = self.shape
        out = np.zeros((n_row, n_col), dtype=float)

        mean_arr = np.asarray(mean)
        cov_arr = np.asarray(cov)
        for i in range(n_row):
            xi = x[i] if x.ndim == 2 else x
            mi = mean_arr[i] if mean_arr.ndim == 2 else mean_arr
            ci = cov_arr[i] if cov_arr.ndim == 3 else cov_arr
            lp = multivariate_normal.logpdf(xi, mean=mi, cov=ci)
            out[i, :] = lp
        return out

    def _ppf(self, p):
        """Inverse transform for multivariate normal.
        
        Parameters
        ----------
        p : 2D np.ndarray same shape as self (n_rows, n_cols)
            or scalar/array broadcastable

        Returns
        -------
        2D np.ndarray same shape as self
        """
        p = np.asarray(p)

        if self.ndim == 0:
            mu = self._bc_params["mean"]
            cov = self._bc_params["cov"]
            return norm.ppf(p, loc=float(mu), scale=float(np.sqrt(cov)))

        n_row, n_col = self.shape

        if p.ndim == 0:
            p_arr = np.full((n_row, n_col), p)
        elif p.ndim == 1:
            if p.shape[0] == n_col:
                p_arr = np.tile(p, (n_row, 1))
            else:
                p_arr = np.broadcast_to(p, (n_row, n_col))

        else:
            p_arr = p

        z = norm.ppf(p_arr)

        mean_arr = np.asarray(self._bc_params["mean"])
        cov_arr = np.asarray(self._bc_params["cov"])

        out = np.zeros((n_row, n_col), dtype=float)
        for i in range(n_row):
            zi = z[i] if z.ndim == 2 else z
            mi = mean_arr[i] if mean_arr.ndim == 2 else mean_arr
            ci = cov_arr[i] if cov_arr.ndim == 3 else cov_arr
            ci = np.asarray(ci)
            try:
                L = np.linalg.cholesky(ci)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(ci)
                vals[vals < 0] = 0.0
                L = vecs @ np.diag(np.sqrt(vals))
            xi = mi + L @ zi
            out[i, :] = xi

        return out

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        idx = pd.RangeIndex(2)
        cols = pd.Index(["x", "y"])
        mean_arr = [[0.0, 1.0], [2.0, 3.0]]
        cov_arr = np.array([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]])

        params1 = {"mean": mean_arr, "cov": cov_arr, "index": idx, "columns": cols}

        params2 = {"mean": [0.0, 0.0], "cov": [[1.0, 0.0], [0.0, 1.0]], "columns": cols}

        # scalar / 1D case (degenerate multivariate of dim 1)
        params3 = {"mean": 0.0, "cov": 1.0}

        return [params1, params2, params3]
