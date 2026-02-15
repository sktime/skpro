# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Kernel mixture distribution (kernel density estimate)."""

__author__ = ["amaydixit11"]

import numpy as np
import pandas as pd
from scipy.special import erf

from skpro.distributions.base import BaseDistribution



# Kernel variance lookup: Var[K] for each built-in kernel
# used in _var to compute Var[X] = h^2 * Var[K] + Var_w(support)
_KERNEL_VARIANCE = {
    "gaussian": 1.0,
    "epanechnikov": 1.0 / 5.0,
    "tophat": 1.0 / 3.0,
    "cosine": 1.0 - 8.0 / (np.pi**2),
    "linear": 1.0 / 6.0,
}



class KernelMixture(BaseDistribution):
    r"""Kernel mixture distribution, a.k.a. kernel density estimate.

    This distribution represents a smooth nonparametric density estimate
    as a weighted mixture of kernel functions centered at support points:

    .. math::

        f(x) = \sum_{i=1}^{n} w_i \frac{1}{h} K\!\left(\frac{x - x_i}{h}\right)

    where :math:`K` is the kernel function, :math:`h` is the bandwidth,
    :math:`x_i` are the support points, and :math:`w_i` are the weights
    (summing to 1).

    This is a vectorized special case of ``Mixture`` where all components
    share a common kernel type and bandwidth. Unlike ``Mixture``, it does
    not create per-component distribution objects, making it efficient
    for large numbers of support points (e.g., kernel density estimation).

    Parameters
    ----------
    support : array-like, 1D
        Support points (data) on which the kernel density is centered.
    bandwidth : float, or str ``"scott"`` or ``"silverman"``, default=1.0
        Bandwidth of the kernel.
        If float, used directly as the bandwidth parameter ``h``.
        If ``"scott"``, bandwidth is computed as
        ``n**(-1/5) * std(support, ddof=1)``.
        If ``"silverman"``, bandwidth is computed as
        ``(4/(3*n))**(1/5) * std(support, ddof=1)``.
    kernel : str or ``BaseDistribution``, default="gaussian"
        The kernel function to use.
        If str, must be one of the built-in kernels:
        ``"gaussian"``, ``"epanechnikov"``, ``"tophat"``,
        ``"cosine"``, ``"linear"``.
        If a ``BaseDistribution`` instance, it is used as a zero-centered,
        unit-scale kernel. The distribution must be scalar (0D).
        This is an experimental feature for custom kernels.
    weights : array-like or None, default=None
        Weights for each support point. If None, uniform weights are used.
        Weights are normalized to sum to 1.
    random_state : int, np.random.Generator, or None, default=None
        Controls randomness for reproducible sampling.
        If int, used as seed for ``np.random.default_rng``.
        If ``np.random.Generator``, used directly.
        If None, a fresh unseeded ``default_rng()`` is created at init.

        .. note::

            When ``kernel`` is a ``BaseDistribution`` instance, the kernel's
            own RNG is used for noise generation and is **not** controlled
            by ``random_state``.  Only the support-point selection is
            reproducible in that case.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.kernel_mixture import KernelMixture
    >>> import numpy as np

    Scalar distribution with built-in Gaussian kernel:

    >>> support = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    >>> km = KernelMixture(support=support, bandwidth=0.5, kernel="gaussian")
    >>> km.mean()
    2.0
    >>> pdf_val = km.pdf(1.5)

    Using a distribution object as a custom kernel (experimental):

    >>> from skpro.distributions.normal import Normal
    >>> km_custom = KernelMixture(
    ...     support=[0.0, 1.0, 2.0],
    ...     bandwidth=0.5,
    ...     kernel=Normal(mu=0, sigma=1),
    ... )

    See Also
    --------
    Mixture : Mixture of arbitrary distribution objects.
    Empirical : Empirical distribution (weighted sum of deltas).
    """

    _tags = {
        "capabilities:approx": ["energy", "ppf", "pdfnorm"],
        "capabilities:exact": ['mean', 'var', 'pdf', 'log_pdf', 'cdf'],
        "distr:measuretype": "continuous",
        "distr:paramtype": "nonparametric",
        "broadcast_init": "off",
    }

    _VALID_KERNELS = {"gaussian", "epanechnikov", "tophat", "cosine", "linear"}

    def __init__(
        self,
        support,
        bandwidth=1.0,
        kernel="gaussian",
        weights=None,
        random_state=None,
        index=None,
        columns=None,
    ):
        self.support = support
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.weights = weights
        self.random_state = random_state

        # resolve RNG once so repeated sample() calls produce independent draws
        if isinstance(random_state, np.random.Generator):
            self._rng = random_state
        elif random_state is not None:
            self._rng = np.random.default_rng(random_state)
        else:
            self._rng = np.random.default_rng()

        # determine kernel mode: "builtin" string or "distribution" object
        if isinstance(kernel, str):
            if kernel not in self._VALID_KERNELS:
                raise ValueError(
                    f"Unknown kernel '{kernel}'. "
                    f"Must be one of {sorted(self._VALID_KERNELS)}."
                )
            self._kernel_mode = "builtin"
        elif isinstance(kernel, BaseDistribution):
            self._kernel_mode = "distribution"
        else:
            raise TypeError(
                f"kernel must be a string or a BaseDistribution instance, "
                f"got {type(kernel).__name__}."
            )

        # coerce support to 1D numpy array
        self._support = np.asarray(support, dtype=float).ravel()
        n = len(self._support)

        # resolve bandwidth
        if isinstance(bandwidth, str):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                std = np.std(self._support, ddof=1)
            if np.isnan(std) or std < 1e-15:
                std = 1.0
            if bandwidth == "scott":
                self._bandwidth = n ** (-1.0 / 5.0) * std
            elif bandwidth == "silverman":
                self._bandwidth = (4.0 / (3.0 * n)) ** (1.0 / 5.0) * std
            else:
                raise ValueError(
                    f"Unknown bandwidth rule '{bandwidth}'. "
                    "Must be a float, 'scott', or 'silverman'."
                )
        else:
            self._bandwidth = float(bandwidth)
            if self._bandwidth <= 0:
                raise ValueError(
                    f"bandwidth must be positive, got {self._bandwidth}."
                )

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
            if np.any(w < 0):
                raise ValueError("All weights must be non-negative.")
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
        if self._kernel_mode == "distribution":
            kernel_dist = self.kernel
            u_arr = np.asarray(u)
            original_shape = u_arr.shape
            u_flat = u_arr.ravel()
            pdf_vals = np.array(
                [kernel_dist._pdf(np.atleast_2d(v)) for v in u_flat],
                dtype=float,
            ).ravel()
            return pdf_vals.reshape(original_shape)

        kernel = self.kernel
        if kernel == "gaussian":
            return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)
        elif kernel == "epanechnikov":
            return np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
        elif kernel == "tophat":
            return np.where(np.abs(u) <= 1, 0.5, 0.0)
        elif kernel == "cosine":
            return np.where(
                np.abs(u) <= 1,
                (np.pi / 4) * np.cos(np.pi * u / 2),
                0.0,
            )
        elif kernel == "linear":
            return np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)
        else:
            raise ValueError(f"Unsupported kernel '{kernel}'.")

    def _kernel_cdf(self, u):
        """Evaluate kernel cdf, vectorized."""
        if self._kernel_mode == "distribution":
            kernel_dist = self.kernel
            u_arr = np.asarray(u)
            original_shape = u_arr.shape
            u_flat = u_arr.ravel()
            cdf_vals = np.array(
                [kernel_dist._cdf(np.atleast_2d(v)) for v in u_flat],
                dtype=float,
            ).ravel()
            return cdf_vals.reshape(original_shape)

        kernel = self.kernel
        if kernel == "gaussian":
            return 0.5 * (1 + erf(u / np.sqrt(2)))
        elif kernel == "epanechnikov":
            cdf_inner = 0.5 + 0.75 * u - 0.25 * u**3
            return np.where(u < -1, 0.0, np.where(u > 1, 1.0, cdf_inner))
        elif kernel == "tophat":
            cdf_inner = 0.5 * (1 + u)
            return np.where(u < -1, 0.0, np.where(u > 1, 1.0, cdf_inner))
        elif kernel == "cosine":
            cdf_inner = 0.5 + 0.5 * np.sin(np.pi * u / 2)
            return np.where(u < -1, 0.0, np.where(u > 1, 1.0, cdf_inner))
        elif kernel == "linear":
            cdf_low = 0.5 * (1 + u) ** 2
            cdf_high = 1 - 0.5 * (1 - u) ** 2
            cdf_inner = np.where(u <= 0, cdf_low, cdf_high)
            return np.where(u < -1, 0.0, np.where(u > 1, 1.0, cdf_inner))
        else:
            raise ValueError(f"Unsupported kernel '{kernel}'.")

    def _kernel_sample(self, size, rng):
        """Sample from the kernel distribution."""
        if self._kernel_mode == "distribution":
            kernel_dist = self.kernel
            n_total = int(np.prod(size)) if isinstance(size, tuple) else size
            samples_df = kernel_dist.sample(n_total)
            samples = samples_df.values.ravel()
            if isinstance(size, tuple):
                return samples.reshape(size)
            return samples
        kernel = self.kernel

        if kernel == "gaussian":
            return rng.standard_normal(size)
        elif kernel == "tophat":
            return rng.uniform(-1, 1, size)
        elif kernel == "epanechnikov":
            return self._rejection_sample(
                size, lambda u: 1 - u**2, rng
            )
        elif kernel == "cosine":
            return self._rejection_sample(
                size, lambda u: np.cos(np.pi * u / 2), rng
            )
        elif kernel == "linear":
            return rng.triangular(-1, 0, 1, size)
        else:
            raise ValueError(f"Unsupported kernel '{kernel}'.")

    @staticmethod
    def _rejection_sample(size, accept_fn, rng):
        """Rejection-sample from uniform(-1,1) with given acceptance function."""
        n_total = int(np.prod(size)) if isinstance(size, tuple) else size
        samples = np.empty(n_total)
        count = 0
        while count < n_total:
            proposal = rng.uniform(-1, 1, n_total - count)
            accept_prob = accept_fn(proposal)
            accepted = proposal[rng.random(n_total - count) < accept_prob]
            n_accept = min(len(accepted), n_total - count)
            samples[count : count + n_accept] = accepted[:n_accept]
            count += n_accept
        return samples.reshape(size) if isinstance(size, tuple) else samples

    def _kernel_variance(self):
        """Return the variance of the kernel function."""
        if self._kernel_mode == "distribution":
            kernel_dist = self.kernel
            var_val = kernel_dist.var()
            if hasattr(var_val, "values"):
                return float(var_val.values.ravel()[0])
            return float(var_val)

        return _KERNEL_VARIANCE[self.kernel]

    def _mean(self):
        r"""Return expected value of the distribution.

        For a kernel mixture:
        :math:`\mathbb{E}[X] = \sum_i w_i x_i`
        """
        mean_val = np.sum(self._weights * self._support)
        if self.ndim > 0:
            return np.full(self.shape, mean_val)
        return mean_val

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        For a kernel mixture, by the law of total variance:
        :math:`\mathrm{Var}[X] = h^2 \mathrm{Var}[K] + \sum_i w_i (x_i - \mu)^2`
        """
        h = self._bandwidth
        mean_val = np.sum(self._weights * self._support)
        weighted_var = np.sum(self._weights * (self._support - mean_val) ** 2)
        kernel_var = self._kernel_variance()
        var_val = h**2 * kernel_var + weighted_var
        if self.ndim > 0:
            return np.full(self.shape, var_val)
        return var_val

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

    def _log_pdf(self, x):
        """Logarithmic probability density function."""
        pdf_val = self._pdf(x)
        if np.isscalar(pdf_val):
            pdf_val = max(pdf_val, 1e-300)
        else:
            pdf_val = np.clip(pdf_val, 1e-300, None)
        return np.log(pdf_val)

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

    def _sample(self, n_samples=None):
        """Sample from the distribution."""
        rng = self._rng
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
            noise = self._kernel_sample(n_draw, rng)
            samples = centers + h * noise

            if n_samples is None:
                return float(samples[0])
            return pd.DataFrame(samples, columns=self.columns)

        n_rows, n_cols = self.shape
        total = n_draw * n_rows * n_cols

        idx = rng.choice(len(support), size=total, p=weights)
        centers = support[idx]
        noise = self._kernel_sample(total, rng)
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
            random_state=self.random_state,
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
            random_state=self.random_state,
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
        params3 = {
            "support": [-1.0, 0.0, 1.0, 2.0],
            "bandwidth": 0.8,
            "kernel": "epanechnikov",
            "weights": [0.1, 0.4, 0.4, 0.1],
        }
        params4 = {
            "support": np.linspace(-2, 2, 20),
            "bandwidth": "scott",
            "kernel": "tophat",
        }
        params5 = {
            "support": [0.0, 1.0, 2.0],
            "bandwidth": 0.5,
            "kernel": "cosine",
            "index": pd.RangeIndex(2),
            "columns": pd.Index(["x"]),
        }
        return [params1, params2, params3, params4, params5]

