# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Erlang probability distribution."""

__author__ = ["RUPESH-KUMAR01"]

import numpy as np
import pandas as pd
from scipy.stats import erlang

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Erlang(_ScipyAdapter):
    r"""Erlang Distribution.

    Most methods wrap ``scipy.stats.erlang``.

    The Erlang Distribution is parameterized by shape :math:`k`
    and rate :math:`\lambda`, such that the pdf is

    .. math:: f(x) = \frac{x^{k-1}\exp\left(-\lambda x\right) \lambda^{k}}{(k-1)!}

    Parameters
    ----------
    rate : float or array of float (1D or 2D)
        Represents the rate parameter, which is also the inverse of the scale parameter.
    k : int or array of int (1D or 2D), optional, default = 1
        Represents the shape parameter.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.erlang import Erlang

    >>> d = Erlang(rate=[[1, 1], [2, 3], [4, 5]], k=2)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["RUPESH-KUMAR01"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, rate, k=1, index=None, columns=None):
        self.rate = rate
        self.k = k

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self):
        return erlang

    def _get_scipy_param(self):
        rate = self._bc_params["rate"]
        k = self._bc_params["k"]

        return [], {"scale": 1 / rate, "a": k}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For Erlang(rate, k), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \int_0^\infty F(t)(1-F(t))\,dt

        using numerical integration over the CDF.
        """
        from scipy.integrate import quad

        rate = np.asarray(self._bc_params["rate"])
        k = np.asarray(self._bc_params["k"])
        rate_b, k_b = np.broadcast_arrays(rate, k)
        result = np.empty_like(rate_b, dtype=float)

        it = np.nditer(
            [rate_b, k_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for rr, kk, out in it:
            rr_val = float(rr)
            kk_val = float(kk)

            def integrand(t, rr=rr_val, kk=kk_val):
                F = erlang.cdf(t, a=kk, scale=1.0 / rr)
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

        :math:`\mathbb{E}[|X-x|]` for X ~ Erlang(rate, k),
        computed via numerical integration.
        """
        from scipy.integrate import quad

        rate = np.asarray(self._bc_params["rate"])
        k = np.asarray(self._bc_params["k"])
        x_arr = np.asarray(x)
        rate_b, k_b = np.broadcast_arrays(rate, k)
        _, x_b = np.broadcast_arrays(rate_b, x_arr)
        result = np.empty_like(rate_b, dtype=float)

        it = np.nditer(
            [rate_b, k_b, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
        )
        for rr, kk, x0, out in it:
            rr_val = float(rr)
            kk_val = float(kk)
            x0_val = float(x0)

            def integrand(t, rr=rr_val, kk=kk_val, x0=x0_val):
                return abs(t - x0) * erlang.pdf(t, a=kk, scale=1.0 / rr)

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
        # Array case examples
        params1 = {
            "rate": 2.0,
            "k": 3,
            "index": pd.Index([0, 1, 2]),
            "columns": pd.Index(["x", "y"]),
        }
        # Scalar case examples
        params2 = {"rate": 0.8, "k": 2}

        params3 = {"rate": 3.0, "k": 1}

        return [params1, params2, params3]
