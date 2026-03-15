# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Binomial probability distribution."""

import numpy as np
import pandas as pd
from scipy.stats import binom, rv_discrete

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Binomial(_ScipyAdapter):
    r"""Binomial distribution.

    Most methods wrap ``scipy.stats.binom``.
    The Binomial distribution is parameterized by the number of trials :math:`n`
    and the probability of success :math:`p`,
    such that the probability mass function (PMF) is given by:

    .. math:: P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}

    Parameters
    ----------
    n : int or array of int (1D or 2D), must be non-negative
    p : float or array of float (1D or 2D), must be in [0, 1]
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.binomial import Binomial

    >>> d = Binomial(n=[[10, 10], [20, 30], [40, 50]], p=0.5)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["meraldoantonio"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pmf"],
        "capabilities:exact": ["mean", "var", "energy", "pmf", "log_pmf", "cdf", "ppf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, n, p, index=None, columns=None):
        self.n = n
        self.p = p

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_discrete:
        return binom

    def _get_scipy_param(self):
        n = self._bc_params["n"]
        p = self._bc_params["p"]

        return [], {"n": n, "p": p}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For Binomial(n, p), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \sum_{k=0}^{n} F(k)(1-F(k))

        where the sum is over the finite integer support 0..n.
        """
        n_arr = np.asarray(self._bc_params["n"], dtype=int)
        p_arr = np.asarray(self._bc_params["p"])
        n_b, p_b = np.broadcast_arrays(n_arr, p_arr)
        result = np.empty_like(p_b, dtype=float)

        it = np.nditer(
            [n_b, p_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for nn, pp, out in it:
            nn_val = int(nn)
            pp_val = float(pp)
            ks = np.arange(0, nn_val + 1)
            Fk = binom.cdf(ks, nn_val, pp_val)
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

        :math:`\mathbb{E}[|X-x|]` for X ~ Binomial(n, p),
        computed as :math:`\sum_{k=0}^{n} |k-x| P(X=k)`.
        """
        n_arr = np.asarray(self._bc_params["n"], dtype=int)
        p_arr = np.asarray(self._bc_params["p"])
        x_arr = np.asarray(x)
        n_b, p_b = np.broadcast_arrays(n_arr, p_arr)
        _, x_b = np.broadcast_arrays(p_b, x_arr)
        result = np.empty_like(p_b, dtype=float)

        it = np.nditer(
            [n_b, p_b, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
        )
        for nn, pp, x0, out in it:
            nn_val = int(nn)
            pp_val = float(pp)
            x0_val = float(x0)
            ks = np.arange(0, nn_val + 1)
            pmf_k = binom.pmf(ks, nn_val, pp_val)
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
        params1 = {"n": [[10, 10], [20, 30], [40, 50]], "p": 0.5}
        params2 = {
            "n": 10,
            "p": 0.5,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # scalar case examples
        params3 = {"n": 15, "p": 0.7}

        return [params1, params2, params3]
