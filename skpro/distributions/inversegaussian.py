# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Inverse Gaussian probability distribution."""

__author__ = ["Omswastik-11"]

import pandas as pd
from scipy.stats import invgauss, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class InverseGaussian(_ScipyAdapter):
    r"""Inverse Gaussian distribution, aka Wald distribution.

    Most methods wrap ``scipy.stats.invgauss``.

    The Inverse Gaussian distribution (Wald) when using SciPy's
    parameterization is specified by a shape parameter ``mu`` and a
    ``scale`` parameter. In SciPy these are the positional and keyword
    parameters of ``scipy.stats.invgauss(mu, scale=scale)``. The
    mean of the distribution is given by ``mean = mu * scale``.

    The pdf in terms of :math:`\mu` = ``mu`` and :math:`\sigma` = ``scale`` is:

    .. math:: f(x; \mu, \sigma) = \sqrt{\frac{\sigma}{2 \pi x^3}}
              \exp\left(-\frac{(x - \mu \sigma)^2}{2 \mu^2 \sigma x}\right)

    Parameters
    ----------
    mu : float or array of float (1D or 2D), must be positive
        shape parameter (dimensionless)
    scale : float or array of float (1D or 2D), must be positive
        scale parameter (multiplies the distribution)
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.inversegaussian import InverseGaussian

    >>> d = InverseGaussian(mu=1.0, scale=1.0)
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, scale, index=None, columns=None):
        self.mu = mu
        self.scale = scale
        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return invgauss

    def _get_scipy_param(self):
        # Pass parameters directly to scipy.stats.invgauss.
        # SciPy's invgauss accepts a shape parameter `mu` and a keyword  `scale`.
        mu = self._bc_params["mu"]
        scale = self._bc_params["scale"]
        return [mu], {"scale": scale}

    def _energy_self(self):
        """Energy of self, w.r.t. self (expected |X-Y| for i.i.d. X,Y ~ InverseGaussian)."""
        import numpy as np
        from scipy.integrate import quad
        from scipy.stats import invgauss

        mu = np.asarray(self._bc_params["mu"])
        scale = np.asarray(self._bc_params["scale"])
        mu_b, scale_b = np.broadcast_arrays(mu, scale)
        result = np.empty_like(mu_b, dtype=float)
        it = np.nditer(
            [mu_b, scale_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for m, s, out in it:

            m_val = m.item()
            s_val = s.item()
            def cdf(x, m_val=m_val, s_val=s_val):
                return invgauss.cdf(x, mu=m_val, scale=s_val)
            def integrand(x, cdf=cdf):
                F = cdf(x)
                return 2 * F * (1 - F)
            val, _ = quad(integrand, 0, np.inf, limit=200)
            out[...] = val
        # Always flatten to 1D of length n_rows for DataFrame compatibility
        n_rows = 1 if self.index is None else len(self.index)
        result = np.asarray(result).reshape(-1)
        if result.shape[0] != n_rows:
            result = result.reshape(n_rows, -1).mean(axis=1)
        if self.index is None and n_rows == 1:
            return result.item()
        return result

    def _energy_x(self, x):
        """Energy of self, w.r.t. a constant frame x (expected |X-x| for X ~ InverseGaussian)."""
        import numpy as np
        from scipy.integrate import quad
        from scipy.stats import invgauss

        mu = np.asarray(self._bc_params["mu"])
        scale = np.asarray(self._bc_params["scale"])
        x = np.asarray(x)
        mu_b, scale_b, x_b = np.broadcast_arrays(mu, scale, x)
        result = np.empty_like(mu_b, dtype=float)
        it = np.nditer(
            [mu_b, scale_b, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
        )
        for m, s, x0, out in it:

            m_val = m.item()
            s_val = s.item()
            x0_val = x0.item()
            def integrand(t, m_val=m_val, s_val=s_val, x0_val=x0_val):
                return np.abs(t - x0_val) * invgauss.pdf(t, mu=m_val, scale=s_val)
            val, _ = quad(integrand, 0, np.inf, limit=200)
            out[...] = val
        # Always flatten to 1D of length n_rows for DataFrame compatibility
        n_rows = 1 if self.index is None else len(self.index)
        result = np.asarray(result).reshape(-1)
        if result.shape[0] != n_rows:
            result = result.reshape(n_rows, -1).mean(axis=1)
        if self.index is None and n_rows == 1:
            return result.item()
        return result

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"mu": [2, 3.5], "scale": [[1, 1], [2, 3], [4, 5]]}
        params2 = {
            "mu": 2.5,
            "scale": 1.5,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"mu": 3.0, "scale": 2.0}

        return [params1, params2, params3]
