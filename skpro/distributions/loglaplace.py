# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Log-Laplace probability distribution."""

import numpy as np
import pandas as pd
from scipy.stats import loglaplace, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class LogLaplace(_ScipyAdapter):
    r"""Log-Laplace distribution.

    Most methods wrap ``scipy.stats.loglaplace``.

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The log-Laplace distribution is a continuous probability distribution obtained by
    taking the logarithm of the Laplace distribution, commonly used in finance and
    hydrology due to its heavy tails and asymmetry.

    The log-Laplace distribution is parametrized by the scale parameter
    :math:`\c`, such that the pdf is

    .. math:: f(x) = \frac{c}{2} x^{c-1}, \quad 0<x<1

    and

    .. math:: f(x) = \frac{c}{2} x^{-c-1}, \quad x >= 1

    The scale parameter :math:`c` is represented by the parameter ``c``.

    Parameters
    ----------
    scale : float or array of float (1D or 2D), must be positive
        scale parameter of the log-Laplace distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.loglaplace import LogLaplace

    >>> ll = LogLaplace(scale=1)
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

    def __init__(self, scale, index=None, columns=None):
        self.scale = scale

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return loglaplace

    def _get_scipy_param(self):
        scale = self._bc_params["scale"]
        return [scale], {}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For LogLaplace(scale), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \int_0^\infty F(t)(1-F(t))\,dt

        using numerical integration.
        """
        from scipy.integrate import quad

        scale = np.asarray(self._bc_params["scale"])
        result = np.empty_like(scale, dtype=float)

        it = np.nditer(
            [scale, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["writeonly"]],
        )
        for ss, out in it:
            ss_val = float(ss)

            def integrand(t, ss=ss_val):
                F = loglaplace.cdf(t, c=ss)
                return 2 * F * (1 - F)

            val, _ = quad(integrand, 0, np.inf, limit=200)
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

        :math:`\mathbb{E}[|X-x|]` for X ~ LogLaplace(scale),
        computed via numerical integration.
        """
        from scipy.integrate import quad

        scale = np.asarray(self._bc_params["scale"])
        x_arr = np.asarray(x)
        scale_b, x_b = np.broadcast_arrays(scale, x_arr)
        result = np.empty_like(scale_b, dtype=float)

        it = np.nditer(
            [scale_b, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for ss, x0, out in it:
            ss_val = float(ss)
            x0_val = float(x0)

            def integrand(t, ss=ss_val, x0=x0_val):
                return abs(t - x0) * loglaplace.pdf(t, c=ss)

            val, _ = quad(integrand, 0, np.inf, limit=200)
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
        params1 = {"scale": [[1, 2], [3, 4]]}
        params2 = {
            "scale": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"scale": 2}
        return [params1, params2, params3]
