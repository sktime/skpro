# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Geometric probability distribution."""

import numpy as np
import pandas as pd
from scipy.stats import geom, rv_discrete

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Geometric(_ScipyAdapter):
    r"""Geometric Distribution.

    Most methods wrap ``scipy.stats.geom``.

    The Geometric distribution is parameterized by the probability of
    success :math:`p` in a given trial
    such that the probability mass function (PMF) is given by:

    .. math:: P(X = k) = p(1 - p)^{k - 1} \quad \text{where} \quad k = 1, 2, 3, \ldots

    Parameters
    ----------
    p : float or array of float (1D or 2D), must be in (0, 1]
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.geometric import Geometric
    >>> d = Geometric(p=0.5)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["aryabhatta-dey"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pmf"],
        "capabilities:exact": ["mean", "var", "energy", "pmf", "log_pmf", "cdf", "ppf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, p, index=None, columns=None):
        self.p = p

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_discrete:
        return geom

    def _get_scipy_param(self):
        p = self._bc_params["p"]

        return [], {"p": p}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For Geometric(p), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \sum_{k=1}^{k_{\max}} F(k)(1-F(k))

        where :math:`k_{\max}` is chosen via the 0.9999999 quantile.
        Support is {1, 2, 3, ...}.
        """
        p_arr = np.asarray(self._bc_params["p"])
        result = np.empty_like(p_arr, dtype=float)

        it = np.nditer(
            [p_arr, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["writeonly"]],
        )
        for pp, out in it:
            pp_val = float(pp)
            k_max = int(geom.ppf(0.9999999, pp_val))
            ks = np.arange(1, k_max + 1)
            Fk = geom.cdf(ks, pp_val)
            out[...] = 2.0 * np.sum(Fk * (1.0 - Fk))

        result_flat = np.asarray(result).reshape(-1)
        n_rows = 1 if self.index is None else len(self.index)
        if result_flat.shape[0] != n_rows:
            result_flat = result_flat.reshape(n_rows, -1).sum(axis=1)
        if self.index is None and n_rows == 1:
            return float(result_flat[0])
        return result_flat

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        :math:`\mathbb{E}[|X-x|]` for X ~ Geometric(p),
        computed as :math:`\sum_{k=1}^{k_{\max}} |k-x| P(X=k)`.
        """
        p_arr = np.asarray(self._bc_params["p"])
        x_arr = np.asarray(x)
        _, x_b = np.broadcast_arrays(p_arr, x_arr)
        result = np.empty_like(p_arr, dtype=float)

        it = np.nditer(
            [p_arr, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for pp, x0, out in it:
            pp_val = float(pp)
            x0_val = float(x0)
            k_max = int(geom.ppf(0.9999999, pp_val))
            ks = np.arange(1, k_max + 1)
            pmf_k = geom.pmf(ks, pp_val)
            out[...] = np.sum(np.abs(ks - x0_val) * pmf_k)

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
        params1 = {"p": [0.2, 0.5, 0.8]}
        params2 = {
            "p": 0.4,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }

        # scalar case examples
        params3 = {"p": 0.7}

        return [params1, params2, params3]
