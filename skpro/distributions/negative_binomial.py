# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Negative binomial probability distribution."""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import nbinom, rv_discrete

from skpro.distributions.adapters.scipy import _ScipyAdapter


class NegativeBinomial(_ScipyAdapter):
    """Negative binomial distribution.

    Most methods wrap ``scipy.stats.nbinom``.

    Parameters
    ----------
    mu : ArrayLike
        mean of the distribution.
    alpha: ArrayLike
        dispersion of distribution.

    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions import NegativeBinomial

    >>> distr = NegativeBinomial(mu=1.0, alpha=1.0)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["tingiskhan"],
        # estimator tags
        # --------------
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "energy", "pmf", "log_pmf", "cdf", "ppf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu: ArrayLike, alpha: ArrayLike, index=None, columns=None):
        self.mu = mu
        self.alpha = alpha

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_discrete:
        return nbinom

    def _get_scipy_param(self) -> dict:
        mu = self._bc_params["mu"]
        alpha = self._bc_params["alpha"]

        n = alpha
        p = alpha / (alpha + mu)

        return [n, p], {}

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For NegativeBinomial(mu, alpha), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \sum_{k=0}^{k_{\max}} F(k)(1-F(k))

        where n=alpha, p=alpha/(alpha+mu) in the nbinom parametrization.
        """
        mu_arr = np.asarray(self._bc_params["mu"])
        alpha_arr = np.asarray(self._bc_params["alpha"])
        mu_b, alpha_b = np.broadcast_arrays(mu_arr, alpha_arr)
        result = np.empty_like(mu_b, dtype=float)

        it = np.nditer(
            [mu_b, alpha_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for mm, aa, out in it:
            mm_val = float(mm)
            aa_val = float(aa)
            n_nb = aa_val
            p_nb = aa_val / (aa_val + mm_val)
            k_max = int(nbinom.ppf(0.9999999, n_nb, p_nb))
            ks = np.arange(0, k_max + 1)
            Fk = nbinom.cdf(ks, n_nb, p_nb)
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

        :math:`\mathbb{E}[|X-x|]` for X ~ NegativeBinomial(mu, alpha),
        computed as :math:`\sum_{k=0}^{k_{\max}} |k-x| P(X=k)`.
        """
        mu_arr = np.asarray(self._bc_params["mu"])
        alpha_arr = np.asarray(self._bc_params["alpha"])
        x_arr = np.asarray(x)
        mu_b, alpha_b = np.broadcast_arrays(mu_arr, alpha_arr)
        _, x_b = np.broadcast_arrays(mu_b, x_arr)
        result = np.empty_like(mu_b, dtype=float)

        it = np.nditer(
            [mu_b, alpha_b, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
        )
        for mm, aa, x0, out in it:
            mm_val = float(mm)
            aa_val = float(aa)
            x0_val = float(x0)
            n_nb = aa_val
            p_nb = aa_val / (aa_val + mm_val)
            k_max = int(nbinom.ppf(0.9999999, n_nb, p_nb))
            ks = np.arange(0, k_max + 1)
            pmf_k = nbinom.pmf(ks, n_nb, p_nb)
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
        params1 = {"mu": [[1, 1], [2, 3], [4, 5]], "alpha": 2.0}
        params2 = {
            "mu": 1.0,
            "alpha": 2.0,
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        return [params1, params2]
