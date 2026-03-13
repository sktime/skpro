# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Half-Logistic probability distribution."""

import numpy as np
import pandas as pd
from scipy.stats import halflogistic, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class HalfLogistic(_ScipyAdapter):
    r"""Half-Logistic distribution.

    Most methods wrap ``scipy.stats.halflogistic``.

    This distribution is univariate, without correlation between dimensions
    for the array-valued case.

    The half-logistic distribution is a continuous probability distribution derived
    from the logistic distribution by taking only the positive half. It is particularly
    useful in reliability analysis, lifetime modeling, and other applications where
    non-negative values are required.

    The half-logistic distribution is parametrized by the scale parameter
    :math:`\beta`, such that the pdf is

    .. math::

        f(x) = \frac{2 \exp\left(-\frac{x}{\beta}\right)}
                {\beta \left(1 + \exp\left(-\frac{x}{\beta}\right)\right)^2},
                x>0 otherwise 0

    The scale parameter :math:`\beta` is represented by the parameter ``beta``.

    Parameters
    ----------
    beta : float or array of float (1D or 2D), must be positive
        scale parameter of the half-logistic distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.halflogistic import HalfLogistic

    >>> hl = HalfLogistic(beta=1)
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

    def __init__(self, beta, index=None, columns=None):
        self.beta = beta

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return halflogistic

    def _get_scipy_param(self):
        beta = self._bc_params["beta"]
        return [beta], {}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For HalfLogistic(beta), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \int_0^\infty F(t)(1-F(t))\,dt

        using numerical integration over the half-logistic CDF.
        ``beta`` acts as the first positional (loc) parameter as in scipy.
        """
        from scipy.integrate import quad

        beta = np.asarray(self._bc_params["beta"])
        result = np.empty_like(beta, dtype=float)

        it = np.nditer(
            [beta, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["writeonly"]],
        )
        for bb, out in it:
            bb_val = float(bb)

            def integrand(t, bb=bb_val):
                F = halflogistic.cdf(t, bb)  # positional, matches _get_scipy_param
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

        :math:`\mathbb{E}[|X-x|]` for X ~ HalfLogistic(beta),
        computed via numerical integration.
        ``beta`` acts as the first positional (loc) parameter as in scipy.
        """
        from scipy.integrate import quad

        beta = np.asarray(self._bc_params["beta"])
        x_arr = np.asarray(x)
        beta_b, x_b = np.broadcast_arrays(beta, x_arr)
        result = np.empty_like(beta_b, dtype=float)

        it = np.nditer(
            [beta_b, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for bb, x0, out in it:
            bb_val = float(bb)
            x0_val = float(x0)

            def integrand(t, bb=bb_val, x0=x0_val):
                return abs(t - x0) * halflogistic.pdf(t, bb)  # positional

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
        params1 = {"beta": [[1, 2], [3, 4]]}
        params2 = {
            "beta": 1,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"beta": 2}
        return [params1, params2, params3]
