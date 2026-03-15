# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Half-Normal probability distribution."""

import numpy as np
import pandas as pd
from scipy.stats import halfnorm, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class HalfNormal(_ScipyAdapter):
    r"""Half-Normal distribution.

    Most methods wrap ``scipy.stats.halfnorm``.

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The distribution is `cut off` at :math:`\( x = 0 \)`. There is no mass assigned to
    negative values; they are entirely excluded from the distribution.

    The half-normal distribution is parametrized by the standard deviation
    :math:`\sigma`, such that the pdf is

    .. math:: f(x) = \frac{\sqrt{2}}{\sigma \sqrt{\pi}}
                    \exp\left(-\frac{x^2}{2\sigma^2}\right), x>0 otherwise 0

    The standard deviation :math:`\sigma` is represented by the parameter ``sigma``.

    Parameters
    ----------
    sigma : float or array of float (1D or 2D), must be positive
        standard deviation of the half-normal distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.halfnormal import HalfNormal

    >>> hn = HalfNormal(sigma=1)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["SaiRevanth25"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, sigma, index=None, columns=None):
        self.sigma = sigma

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return halfnorm

    def _get_scipy_param(self):
        sigma = self._bc_params["sigma"]
        return [sigma], {}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For HalfNormal(sigma), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \int_{\sigma}^\infty F(t)(1-F(t))\,dt

        using numerical integration. The distribution is
        ``halfnorm(loc=sigma, scale=1)`` as passed by ``_get_scipy_param``.
        """
        from scipy.integrate import quad

        sigma = np.asarray(self._bc_params["sigma"])
        result = np.empty_like(sigma, dtype=float)

        it = np.nditer(
            [sigma, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["writeonly"]],
        )
        for ss, out in it:
            ss_val = float(ss)

            def integrand(t, ss=ss_val):
                # positional arg = loc, as in _get_scipy_param: [sigma], {}
                F = halfnorm.cdf(t, ss)
                return 2 * F * (1 - F)

            val, _ = quad(integrand, ss_val, np.inf, limit=200)
            out[...] = val

        result_flat = np.asarray(result).reshape(-1)
        n_rows = 1 if self.index is None else len(self.index)
        if result_flat.shape[0] != n_rows:
            result_flat = result_flat.reshape(n_rows, -1).sum(axis=1)
        if self.index is None and n_rows == 1:
            return float(result_flat[0])
        return result_flat

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        :math:`\mathbb{E}[|X-x|]` for X ~ HalfNormal(sigma),
        computed via numerical integration.
        The distribution is ``halfnorm(loc=sigma, scale=1)``.
        """
        from scipy.integrate import quad

        sigma = np.asarray(self._bc_params["sigma"])
        x_arr = np.asarray(x)
        sigma_b, x_b = np.broadcast_arrays(sigma, x_arr)
        result = np.empty_like(sigma_b, dtype=float)

        it = np.nditer(
            [sigma_b, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for ss, x0, out in it:
            ss_val = float(ss)
            x0_val = float(x0)

            def integrand(t, ss=ss_val, x0=x0_val):
                # positional arg = loc
                return abs(t - x0) * halfnorm.pdf(t, ss)

            val, _ = quad(integrand, ss_val, np.inf, limit=200)
            out[...] = val

        result_flat = np.asarray(result).reshape(-1)
        n_rows = 1 if self.index is None else len(self.index)
        if result_flat.shape[0] != n_rows:
            result_flat = result_flat.reshape(n_rows, -1).sum(axis=1)
        if self.index is None and n_rows == 1:
            return float(result_flat[0])
        return result_flat

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # array case examples
        params1 = {"sigma": [[1, 2], [3, 4]]}
        params2 = {
            "sigma": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"sigma": 2}
        return [params1, params2, params3]
