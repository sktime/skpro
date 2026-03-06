# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Log-logistic aka Fisk probability distribution."""

import numpy as np
import pandas as pd
from scipy.stats import fisk, rv_continuous

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Fisk(_ScipyAdapter):
    r"""Fisk distribution, aka log-logistic distribution.

    Most methods wrap ``scipy.stats.fisk``.

    The Fisk distribution is parametrized by a scale parameter :math:`\alpha`
    and a shape parameter :math:`\beta`, such that the cumulative distribution
    function (CDF) is given by:

    .. math:: F(x) = 1 - \left(1 + \frac{x}{\alpha}\right)^{-\beta}\right)^{-1}

    Parameters
    ----------
    alpha : float or array of float (1D or 2D), must be positive
        scale parameter of the distribution
    beta : float or array of float (1D or 2D), must be positive
        shape parameter of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.fisk import Fisk

    >>> d = Fisk(beta=[[1, 1], [2, 3], [4, 5]], alpha=2)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly", "malikrafsan"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, alpha=1, beta=1, index=None, columns=None):
        self.alpha = alpha
        self.beta = beta

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return fisk

    def _get_scipy_param(self):
        alpha = self._bc_params["alpha"]
        beta = self._bc_params["beta"]

        return [], {"c": beta, "scale": alpha}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For Fisk(alpha, beta) (log-logistic), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \int_0^\infty F(t)(1-F(t))\,dt

        using numerical integration. Requires beta > 1 for finiteness.
        """
        from scipy.integrate import quad

        alpha = np.asarray(self._bc_params["alpha"])
        beta = np.asarray(self._bc_params["beta"])
        alpha_b, beta_b = np.broadcast_arrays(alpha, beta)
        result = np.empty_like(alpha_b, dtype=float)

        it = np.nditer(
            [alpha_b, beta_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for aa, bb, out in it:
            aa_val = float(aa)
            bb_val = float(bb)

            def integrand(t, aa=aa_val, bb=bb_val):
                F = fisk.cdf(t, c=bb, scale=aa)
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

        :math:`\mathbb{E}[|X-x|]` for X ~ Fisk(alpha, beta),
        computed via numerical integration.
        """
        from scipy.integrate import quad

        alpha = np.asarray(self._bc_params["alpha"])
        beta = np.asarray(self._bc_params["beta"])
        x_arr = np.asarray(x)
        alpha_b, beta_b = np.broadcast_arrays(alpha, beta)
        _, x_b = np.broadcast_arrays(alpha_b, x_arr)
        result = np.empty_like(alpha_b, dtype=float)

        it = np.nditer(
            [alpha_b, beta_b, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
        )
        for aa, bb, x0, out in it:
            aa_val = float(aa)
            bb_val = float(bb)
            x0_val = float(x0)

            def integrand(t, aa=aa_val, bb=bb_val, x0=x0_val):
                return abs(t - x0) * fisk.pdf(t, c=bb, scale=aa)

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
        params1 = {"alpha": [[1, 1], [2, 3], [4, 5]], "beta": 3}
        params2 = {
            "alpha": 2,
            "beta": 3,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"alpha": 1.5, "beta": 2.1}

        return [params1, params2, params3]
