# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Kernel mixture distribution (kernel density estimate)."""

__author__ = ["amaydixit11"]

import numpy as np
import pandas as pd
from scipy.special import erf

from skpro.distributions.base import BaseDistribution



class KernelMixture(BaseDistribution):
    r"""Kernel mixture distribution, a.k.a. kernel density estimate.

    This distribution represents a smooth nonparametric density estimate
    as a weighted mixture of kernel functions centered at support points:

    .. math::

        f(x) = \sum_{i=1}^{n} w_i \frac{1}{h} K\!\left(\frac{x - x_i}{h}\right)

    where :math:`K` is the kernel function, :math:`h` is the bandwidth,
    :math:`x_i` are the support points, and :math:`w_i` are the weights
    (summing to 1).

    Parameters
    ----------
    support : array-like, 1D
        Support points (data) on which the kernel density is centered.
    bandwidth : float, default=1.0
        Bandwidth of the kernel.
    kernel : str, default="gaussian"
        The kernel function to use.
    weights : array-like or None, default=None
        Weights for each support point. If None, uniform weights are used.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    See Also
    --------
    Mixture : Mixture of arbitrary distribution objects.
    Empirical : Empirical distribution (weighted sum of deltas).
    """

    _tags = {
        "capabilities:approx": ["energy", "ppf", "pdfnorm"],
        "capabilities:exact": ['pdf', 'cdf'],
        "distr:measuretype": "continuous",
        "distr:paramtype": "nonparametric",
        "broadcast_init": "off",
    }

    _VALID_KERNELS = {"gaussian"}

    def __init__(
        self,
        support,
        bandwidth=1.0,
        kernel="gaussian",
        weights=None,
        index=None,
        columns=None,
    ):
        self.support = support
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.weights = weights

        # validate kernel
        if kernel not in self._VALID_KERNELS:
            raise ValueError(
                f"Unknown kernel '{kernel}'. "
                f"Must be one of {sorted(self._VALID_KERNELS)}."
            )

        # coerce support to 1D numpy array
        self._support = np.asarray(support, dtype=float).ravel()
        n = len(self._support)

        self._bandwidth = float(bandwidth)

        # normalize weights
        if weights is None:
            self._weights = np.ones(n) / n
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if len(w) != n:
                raise ValueError(
                    f"weights length ({len(w)}) must match "
                    f"support length ({n})."
                )
            w_sum = np.sum(w)
            if w_sum <= 0:
                raise ValueError("weights must have positive sum.")
            self._weights = w / w_sum

        super().__init__(index=index, columns=columns)

    def _init_shape_bc(self, index=None, columns=None):
        """Initialize shape — scalar or 2D based on index/columns."""
        if index is None and columns is None:
            self._shape = ()
        else:
            n_rows = len(index) if index is not None else 1
            n_cols = len(columns) if columns is not None else 1
            self._shape = (n_rows, n_cols)

    # --- Kernel evaluation helpers ---

    def _kernel_pdf(self, u):
        """Evaluate kernel pdf K(u), vectorized."""
        kernel = self.kernel
        if kernel == "gaussian":
            return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)

    def _kernel_cdf(self, u):
        """Evaluate kernel cdf, vectorized."""
        kernel = self.kernel
        if kernel == "gaussian":
            return 0.5 * (1 + erf(u / np.sqrt(2)))

    # --- BaseDistribution interface ---

    def _pdf(self, x):
        """Probability density function."""
        h = self._bandwidth
        support = self._support
        weights = self._weights

        if self.ndim == 0:
            x_val = float(x)
            u = (x_val - support) / h
            return np.sum(weights * self._kernel_pdf(u)) / h

        x_flat = x.ravel()
        u = (x_flat[:, None] - support[None, :]) / h
        K = self._kernel_pdf(u)
        pdf_flat = np.sum(weights[None, :] * K, axis=1) / h
        return pdf_flat.reshape(x.shape)

    def _cdf(self, x):
        """Cumulative distribution function."""
        h = self._bandwidth
        support = self._support
        weights = self._weights

        if self.ndim == 0:
            x_val = float(x)
            u = (x_val - support) / h
            return np.sum(weights * self._kernel_cdf(u))

        x_flat = x.ravel()
        u = (x_flat[:, None] - support[None, :]) / h
        K_cdf = self._kernel_cdf(u)
        cdf_flat = np.sum(weights[None, :] * K_cdf, axis=1)
        return cdf_flat.reshape(x.shape)

    def _kernel_sample(self, size):
        """Sample from the kernel distribution."""
        rng = np.random.default_rng()
        return rng.standard_normal(size)

    def _sample(self, n_samples=None):
        """Sample from the distribution."""
        rng = np.random.default_rng()
        h = self._bandwidth
        support = self._support
        weights = self._weights

        if n_samples is None:
            n_draw = 1
        else:
            n_draw = n_samples

        if self.ndim == 0:
            idx = rng.choice(len(support), size=n_draw, p=weights)
            centers = support[idx]
            noise = self._kernel_sample(n_draw)
            samples = centers + h * noise

            if n_samples is None:
                return float(samples[0])
            return pd.DataFrame(samples, columns=self.columns)

        n_rows, n_cols = self.shape
        total = n_draw * n_rows * n_cols

        idx = rng.choice(len(support), size=total, p=weights)
        centers = support[idx]
        noise = self._kernel_sample(total)
        samples_flat = centers + h * noise
        samples = samples_flat.reshape(n_draw, n_rows, n_cols)

        if n_samples is None:
            return pd.DataFrame(
                samples[0], index=self.index, columns=self.columns
            )

        spl_index = pd.MultiIndex.from_product([range(n_draw), self.index])
        return pd.DataFrame(
            samples.reshape(n_draw * n_rows, n_cols),
            index=spl_index,
            columns=self.columns,
        )

    def _iloc(self, rowidx=None, colidx=None):
        """Subset distribution by integer row/column indices."""
        from skpro.distributions.base._base import is_scalar_notnone

        if is_scalar_notnone(rowidx) and is_scalar_notnone(colidx):
            return self._iat(rowidx, colidx)
        if is_scalar_notnone(rowidx):
            rowidx = pd.Index([rowidx])
        if is_scalar_notnone(colidx):
            colidx = pd.Index([colidx])

        index = self.index
        columns = self.columns

        if rowidx is not None:
            index_subset = index.take(rowidx)
        else:
            index_subset = index

        if colidx is not None:
            columns_subset = columns.take(colidx)
        else:
            columns_subset = columns

        return KernelMixture(
            support=self.support,
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            weights=self.weights,
            index=index_subset,
            columns=columns_subset,
        )

    def _iat(self, rowidx=None, colidx=None):
        """Subset distribution to a single scalar element."""
        return KernelMixture(
            support=self.support,
            bandwidth=self.bandwidth,
            kernel=self.kernel,
            weights=self.weights,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "support": [0.0, 1.0, 2.0, 3.0, 4.0],
            "bandwidth": 0.5,
            "kernel": "gaussian",
        }
        params2 = {
            "support": [0.0, 1.0, 2.0, 3.0, 4.0],
            "bandwidth": 1.0,
            "kernel": "gaussian",
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]

