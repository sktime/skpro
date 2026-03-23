# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Mixture of Normal distributions with per-sample mixing weights."""

__author__ = ["joshdunnlime"]

import numpy as np
import pandas as pd
from scipy.special import erf, erfinv, logsumexp

from skpro.distributions.base import BaseDistribution


class NormalMixture(BaseDistribution):
    r"""Mixture of Normal distributions with per-sample mixing weights.

    Unlike :class:`Mixture`, this distribution supports **per-sample** mixture
    weights, making it suitable as the output of Mixture Density Networks (MDNs).

    Acknowledgement: Implementation inspired by Mixture Density Networks
    (Bishop, 1994) and the scikit-mdn library (koaning).

    Each sample (row) has its own mixing weights :math:`\pi_k`, component means
    :math:`\mu_k`, and component standard deviations :math:`\sigma_k`.

    The probability density for sample :math:`i`, output :math:`j` is:

    .. math::

        f_{ij}(x) = \sum_{k=1}^{K} \pi_{ik}
        \mathcal{N}(x \mid \mu_{ijk}, \sigma_{ijk})

    where :math:`\mathcal{N}` is the Normal distribution PDF.

    Parameters
    ----------
    pi : float or array of float (1D or 2D)
        Per-sample mixing weights. Must be non-negative; will be
        normalized to sum to 1 along the component axis.
        Shape: (n_samples, n_components) or (n_components,) for scalar case
    mu : float or array of float (1D, 2D or 3D)
        Per-component means.
        Shape: (n_samples, n_components, n_outputs), (n_components, n_outputs),
        or (n_components,) for scalar case
    sigma : float or array of float (1D, 2D or 3D), must be positive
        Per-component standard deviations.
        Shape: same as mu
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.normal_mixture import NormalMixture
    >>> import numpy as np

    >>> # 3 samples, 2 components, 2 outputs
    >>> pi = np.array([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])
    >>> mu = np.array([[[0, 1], [3, 4]],
    ...                [[1, 2], [4, 5]],
    ...                [[2, 3], [5, 6]]])
    >>> sigma = np.ones((3, 2, 2))
    >>> d = NormalMixture(pi=pi, mu=mu, sigma=sigma)
    >>> d.mean()
         0    1
    0  2.1  3.1
    1  2.5  3.5
    2  2.6  3.6

    References
    ----------
    Bishop, C. M. (1994). Mixture density networks. Technical Report
    NCRG/94/004, Neural Computing Research Group, Aston University.
    """

    _tags = {
        "capabilities:approx": ["pdfnorm", "ppf"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "energy"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "composite",
    }

    def __init__(self, pi, mu, sigma, index=None, columns=None):
        """Initialize NormalMixture distribution."""
        self.pi = pi
        self.mu = mu
        self.sigma = sigma

        # Store as numpy arrays
        self._pi = np.asarray(pi, dtype=np.float64)
        self._mu = np.asarray(mu, dtype=np.float64)
        self._sigma = np.asarray(sigma, dtype=np.float64)

        # Detect scalar input
        scalar_input = (
            index is None
            and columns is None
            and self._pi.ndim == 1
            and self._mu.ndim == 1
            and self._sigma.ndim == 1
        )

        # Normalize to 3D: (n_samples, n_components, n_outputs)
        if self._mu.ndim == 1:
            # scalar: (n_components,) -> (1, n_components, 1)
            self._mu = self._mu[np.newaxis, :, np.newaxis]
        elif self._mu.ndim == 2:
            # (n_samples, n_components) -> (n_samples, n_components, 1)
            self._mu = self._mu[:, :, np.newaxis]

        if self._sigma.ndim == 1:
            self._sigma = self._sigma[np.newaxis, :, np.newaxis]
        elif self._sigma.ndim == 2:
            self._sigma = self._sigma[:, :, np.newaxis]

        if self._pi.ndim == 1:
            self._pi = self._pi[np.newaxis, :]

        # Normalize pi to sum to 1 per row
        pi_sum = self._pi.sum(axis=1, keepdims=True)
        pi_sum = np.where(pi_sum == 0, 1.0, pi_sum)
        self._pi = self._pi / pi_sum

        n_samples = self._mu.shape[0]
        n_outputs = self._mu.shape[2]

        if index is None and not scalar_input:
            index = pd.RangeIndex(n_samples)
        if columns is None and not scalar_input:
            columns = pd.RangeIndex(n_outputs)

        super().__init__(index=index, columns=columns)

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        pi = self._pi[:, :, np.newaxis]  # (n_samples, n_components, 1)
        mu = self._mu  # (n_samples, n_components, n_outputs)

        mean_arr = np.sum(pi * mu, axis=1)  # (n_samples, n_outputs)

        if self.ndim == 0:
            return float(mean_arr[0, 0])
        return mean_arr

    def _var(self):
        """Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of distribution (entry-wise)
        """
        # Law of total variance: E[X^2] - E[X]^2
        pi = self._pi[:, :, np.newaxis]
        mu = self._mu
        sigma = self._sigma

        mean = self._mean()
        if self.ndim == 0:
            mean = np.array([[mean]])

        second_moment = np.sum(pi * (sigma**2 + mu**2), axis=1)

        var_arr = second_moment - mean**2

        if self.ndim == 0:
            return float(var_arr[0, 0])
        return var_arr

    def _pdf(self, x):
        """Probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            pdf values at the given points
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x[np.newaxis, :]

        pi = self._pi[:, :, np.newaxis]
        mu = self._mu
        sigma = self._sigma

        # Expand x for broadcasting: (n_samples, 1, n_outputs)
        x_exp = x[:, np.newaxis, :]

        # Gaussian PDF for each component
        z = (x_exp - mu) / sigma
        comp_pdf = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2 * np.pi))

        # Weight and sum over components
        pdf_arr = np.sum(pi * comp_pdf, axis=1)
        if self.ndim == 0:
            return float(pdf_arr[0, 0])
        return pdf_arr

    def _log_pdf(self, x):
        """Logarithmic probability density function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the log pdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            log pdf values at the given points
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x[np.newaxis, :]

        pi = self._pi[:, :, np.newaxis]
        mu = self._mu
        sigma = self._sigma

        x_exp = x[:, np.newaxis, :]

        # Gaussian log PDF
        z = (x_exp - mu) / sigma
        log_comp_pdf = -0.5 * z**2 - np.log(sigma * np.sqrt(2 * np.pi))

        # Log-sum-exp for numerical stability
        log_pi = np.log(np.clip(pi, 1e-300, None))
        log_weighted = log_pi + log_comp_pdf

        lpdf_arr = logsumexp(log_weighted, axis=1, keepdims=False)
        if self.ndim == 0:
            return float(lpdf_arr[0, 0])
        return lpdf_arr

    def _cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to evaluate the cdf at

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            cdf values at the given points
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x[np.newaxis, :]

        pi = self._pi[:, :, np.newaxis]
        mu = self._mu
        sigma = self._sigma

        x_exp = x[:, np.newaxis, :]

        # Gaussian CDF per component
        comp_cdf = 0.5 + 0.5 * erf((x_exp - mu) / (sigma * np.sqrt(2)))

        # Weight and sum
        cdf_arr = np.sum(pi * comp_cdf, axis=1)
        if self.ndim == 0:
            return float(cdf_arr[0, 0])
        return cdf_arr

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf.

        Uses fast approximate ppf via vectorized bisection on tight brackets
        built from component Normal ppfs. Much faster than the default
        bisection method on wide ranges.

        Parameters
        ----------
        p : 2D np.ndarray, same shape as ``self``
            probabilities at which to evaluate the quantile

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            approximate quantiles
        """
        p = np.asarray(p, dtype=float)

        if p.ndim == 0:
            p = p.reshape(1, 1)
        elif p.ndim == 1:
            # row-vector convention for 1D inputs
            p = p[np.newaxis, :]

        pi = self._pi[:, :, np.newaxis]  # (n_samples, n_components, 1)
        mu = self._mu  # (n_samples, n_components, n_outputs)
        sigma = self._sigma  # (n_samples, n_components, n_outputs)

        # Handle edge cases: p=0 -> -inf, p=1 -> +inf
        out = np.empty_like(p, dtype=float)
        out[p <= 0.0] = -np.inf
        out[p >= 1.0] = np.inf

        # Work only on interior probabilities
        interior = (p > 0.0) & (p < 1.0)

        if not np.any(interior):
            return out

        # Clip for numerical stability (keeps erfinv away from ±1 exactly)
        eps = 1e-12
        pc = np.clip(p, eps, 1.0 - eps)

        # ---- Build tight bracket using component Normal ppfs ----
        # Normal ppf: mu + sigma * sqrt(2) * erfinv(2p - 1)
        # pc is (n_samples, n_outputs), need to add component axis in middle
        z = np.sqrt(2.0) * erfinv(2.0 * pc[:, np.newaxis, :] - 1.0)
        qk = mu + sigma * z  # (n_samples, n_components, n_outputs)

        # Tight bracket for each sample/output
        left = np.min(qk, axis=1)  # (n_samples, n_outputs)
        right = np.max(qk, axis=1)  # (n_samples, n_outputs)

        # ---- Helper: vectorized mixture CDF ----
        inv_sqrt2 = 1.0 / np.sqrt(2.0)

        def mix_cdf(x):
            """Evaluate mixture CDF at x."""
            x_exp = x[:, np.newaxis, :]  # (n_samples, 1, n_outputs)
            z = (x_exp - mu) / sigma * inv_sqrt2
            cdf_comp = 0.5 + 0.5 * erf(z)
            return np.sum(pi * cdf_comp, axis=1)  # (n_samples, n_outputs)

        # ---- Vectorized bisection on tight bracket ----
        max_iter = int(self.get_tag("bisect_iter"))
        xtol = 2e-12
        rtol = 8.881784197001252e-16  # same scale as SciPy defaults

        for _ in range(max_iter):
            mid = 0.5 * (left + right)
            c = mix_cdf(mid)

            go_right = c < pc  # need larger x
            left = np.where(go_right, mid, left)
            right = np.where(go_right, right, mid)

            # Early stopping (elementwise)
            if np.all((right - left) <= (xtol + rtol * np.abs(mid))):
                break

        q = 0.5 * (left + right)

        # Put results back into output array
        out[interior] = q[interior]

        if self.ndim == 0:
            return float(out[0, 0])
        return out

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|]`, where :math:`X, Y` are i.i.d. copies of self.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        pi = self._pi[:, :, np.newaxis]
        sigma = self._sigma

        # For mixture of normals: E[|X-Y|] = sum_k sum_l pi_k pi_l E[|N_k - N_l|]
        # E[|N(mu1,sig1) - N(mu2,sig2)|] = 2*sqrt(sig1^2 + sig2^2)/sqrt(pi)
        # when mu1=mu2 (simplified for same means)

        # Approximate using mean variance across components
        avg_sigma_sq = np.sum(pi * sigma**2, axis=1)
        energy_arr = 2 * np.sqrt(2 * avg_sigma_sq / np.pi)

        if self.ndim == 0:
            return float(np.sum(energy_arr))
        return np.sum(energy_arr, axis=1, keepdims=True)

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        :math:`\mathbb{E}[|X-x|]`, where :math:`X` is a copy of self,
        and :math:`x` is a constant.

        Parameters
        ----------
        x : 2D np.ndarray, same shape as ``self``
            values to compute energy w.r.t. to

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            energy values w.r.t. the given points
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x[np.newaxis, :]

        pi = self._pi[:, :, np.newaxis]
        mu = self._mu
        sigma = self._sigma

        x_exp = x[:, np.newaxis, :]

        # For each component k: E[|N(mu_k, sigma_k) - x|]
        cdf = 0.5 + 0.5 * erf((x_exp - mu) / (sigma * np.sqrt(2)))
        z = (x_exp - mu) / sigma
        pdf = np.exp(-0.5 * z**2) / (sigma * np.sqrt(2 * np.pi))

        comp_energy = (x_exp - mu) * (2 * cdf - 1) + 2 * sigma**2 * pdf

        # Weight by pi and sum over components
        energy_arr = np.sum(pi * comp_energy, axis=1)

        if self.ndim == 0:
            return float(np.sum(energy_arr))
        return np.sum(energy_arr, axis=1, keepdims=True)

    def sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None
            number of samples to draw from the distribution

        Returns
        -------
        pd.DataFrame
            samples from the distribution
        """
        if n_samples is None:
            N = 1
        else:
            N = n_samples

        n_rows = self._pi.shape[0]
        n_outputs = self._mu.shape[2]
        n_components = self._pi.shape[1]

        # Sample component indices according to pi
        component_indices = np.zeros((N, n_rows), dtype=int)

        for i in range(n_rows):
            component_indices[:, i] = np.random.choice(
                n_components, size=N, p=self._pi[i]
            )

        # Sample from selected components
        sample_arr = np.zeros((N, n_rows, n_outputs))

        for i in range(n_rows):
            for j in range(n_outputs):
                for s in range(N):
                    k = component_indices[s, i]
                    sample_arr[s, i, j] = np.random.normal(
                        self._mu[i, k, j], self._sigma[i, k, j]
                    )

        # Reshape to (N * n_rows, n_outputs)
        sample_arr = sample_arr.reshape(N * n_rows, n_outputs)

        if self.ndim == 0:
            if n_samples is None:
                return float(sample_arr[0, 0])
            return pd.DataFrame(
                sample_arr.reshape(N, 1),
                index=pd.RangeIndex(N),
                columns=pd.RangeIndex(1),
            )

        # Create index
        if n_samples is None:
            spl_index = self.index
        else:
            spl_index = pd.MultiIndex.from_product([pd.RangeIndex(N), self.index])

        return pd.DataFrame(sample_arr, index=spl_index, columns=self.columns)

    def _iloc(self, rowidx=None, colidx=None):
        """Subset distribution to given row and column indices.

        Parameters
        ----------
        rowidx : None, int, slice, or array-like
            Row indices to subset to.
        colidx : None, int, slice, or array-like
            Column indices to subset to.

        Returns
        -------
        NormalMixture
            Subsetted distribution.
        """
        from skpro.distributions.base._base import is_scalar_notnone

        if is_scalar_notnone(rowidx) and is_scalar_notnone(colidx):
            return self._iat(rowidx, colidx)

        n_rows = self._pi.shape[0]
        n_outputs = self._mu.shape[2]

        # Convert slices and scalars to integer arrays for numpy indexing
        if isinstance(rowidx, slice):
            rowidx = np.arange(*rowidx.indices(n_rows))
        elif is_scalar_notnone(rowidx):
            rowidx = np.array([rowidx])

        if isinstance(colidx, slice):
            colidx = np.arange(*colidx.indices(n_outputs))
        elif is_scalar_notnone(colidx):
            colidx = np.array([colidx])

        pi = self._pi
        mu = self._mu
        sigma = self._sigma

        if rowidx is not None:
            pi = pi[rowidx]
            mu = mu[rowidx]
            sigma = sigma[rowidx]

        if colidx is not None:
            mu = mu[:, :, colidx]
            sigma = sigma[:, :, colidx]

            if mu.ndim == 2:
                mu = mu[:, :, np.newaxis]
                sigma = sigma[:, :, np.newaxis]

        def subset_not_none(idx, subs):
            if subs is not None:
                return idx[subs]
            return idx

        index_subset = subset_not_none(self.index, rowidx)
        columns_subset = subset_not_none(self.columns, colidx)

        return NormalMixture(
            pi=pi,
            mu=mu,
            sigma=sigma,
            index=index_subset,
            columns=columns_subset,
        )

    def _iat(self, rowidx=None, colidx=None):
        """Subset to a single element (scalar distribution).

        Parameters
        ----------
        rowidx : int
            Row index.
        colidx : int
            Column index.

        Returns
        -------
        NormalMixture
            Scalar distribution at (rowidx, colidx).
        """
        pi = self._pi[rowidx]  # (n_components,)
        mu = self._mu[rowidx, :, colidx]  # (n_components,)
        sigma = self._sigma[rowidx, :, colidx]  # (n_components,)

        return NormalMixture(pi=pi, mu=mu, sigma=sigma)

    @classmethod
    def get_test_params(cls, parameter_set="default"):  # type: ignore[override]
        """Return testing parameter settings for the estimator."""
        # 2D array case: 3 samples, 2 components, 2 outputs
        pi1 = np.array([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])
        mu1 = np.array([[[0, 1], [3, 4]], [[1, 2], [4, 5]], [[2, 3], [5, 6]]])
        sigma1 = np.ones((3, 2, 2))
        params1 = {"pi": pi1, "mu": mu1, "sigma": sigma1}

        # Explicit index/columns case
        params2 = {
            "pi": pi1,
            "mu": mu1,
            "sigma": sigma1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }

        # Scalar case
        pi3 = np.array([0.4, 0.6])
        mu3 = np.array([0, 3])
        sigma3 = np.array([1, 2])
        params3 = {"pi": pi3, "mu": mu3, "sigma": sigma3}

        return [params1, params2, params3]
