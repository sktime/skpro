"""Skellam probability distribution for skpro."""

import numpy as np
from scipy.stats import rv_discrete, skellam

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Skellam(_ScipyAdapter):
    r"""Skellam probability distribution.

    The Skellam distribution is a discrete probability distribution that describes
    the difference between two independent Poisson-distributed random variables
    with means $\mu_1$ and $\mu_2$. Its probability mass function (PMF) is:

    .. math::
        P(k; \mu_1, \mu_2) = e^{-(\mu_1 + \mu_2)}
        \left( \frac{\mu_1}{\mu_2} \right)^{k/2} I_{|k|}(2 \sqrt{\mu_1 \mu_2})

    where $I_{|k|}$ is the modified Bessel function of the first kind.

    Parameters
    ----------
    mu1 : float
        Mean of the first Poisson distribution
    mu2 : float
        Mean of the second Poisson distribution
    """

    _tags = {
        "authors": ["arnavk23"],
        "distr:measuretype": "discrete",
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "energy", "pmf", "log_pmf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, mu1, mu2, index=None, columns=None):
        self.mu1 = mu1
        self.mu2 = mu2
        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_discrete:
        return skellam

    def _get_scipy_param(self):
        mu1 = self._bc_params["mu1"]
        mu2 = self._bc_params["mu2"]
        return [mu1, mu2], {}

    def _pmf(self, x):
        mu1 = self._bc_params["mu1"]
        mu2 = self._bc_params["mu2"]
        return skellam.pmf(x, mu1, mu2)

    def _cdf(self, x):
        mu1 = self._bc_params["mu1"]
        mu2 = self._bc_params["mu2"]
        return skellam.cdf(x, mu1, mu2)

    def _ppf(self, p):
        mu1 = self._bc_params["mu1"]
        mu2 = self._bc_params["mu2"]
        return skellam.ppf(p, mu1, mu2)

    def _mean(self):
        mu1 = self._bc_params["mu1"]
        mu2 = self._bc_params["mu2"]
        return skellam.mean(mu1, mu2)

    def _var(self):
        mu1 = self._bc_params["mu1"]
        mu2 = self._bc_params["mu2"]
        return skellam.var(mu1, mu2)

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For Skellam(mu1, mu2), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \sum_{k=k_{\min}}^{k_{\max}} F(k)(1-F(k))

        where bounds are from ppf(1e-7) to ppf(1-1e-7). Skellam support is ℤ.
        """
        mu1_arr = np.asarray(self._bc_params["mu1"])
        mu2_arr = np.asarray(self._bc_params["mu2"])
        mu1_b, mu2_b = np.broadcast_arrays(mu1_arr, mu2_arr)
        result = np.empty_like(mu1_b, dtype=float)

        it = np.nditer(
            [mu1_b, mu2_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["writeonly"]],
        )
        for m1, m2, out in it:
            m1_val = float(m1)
            m2_val = float(m2)
            k_lo = int(skellam.ppf(1e-7, m1_val, m2_val))
            k_hi = int(skellam.ppf(1 - 1e-7, m1_val, m2_val))
            ks = np.arange(k_lo, k_hi + 1)
            Fk = skellam.cdf(ks, m1_val, m2_val)
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

        :math:`\mathbb{E}[|X-x|]` for X ~ Skellam(mu1, mu2),
        computed as :math:`\sum_k |k-x| P(X=k)` over a truncated integer range.
        """
        mu1_arr = np.asarray(self._bc_params["mu1"])
        mu2_arr = np.asarray(self._bc_params["mu2"])
        x_arr = np.asarray(x)
        mu1_b, mu2_b = np.broadcast_arrays(mu1_arr, mu2_arr)
        _, x_b = np.broadcast_arrays(mu1_b, x_arr)
        result = np.empty_like(mu1_b, dtype=float)

        it = np.nditer(
            [mu1_b, mu2_b, x_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
        )
        for m1, m2, x0, out in it:
            m1_val = float(m1)
            m2_val = float(m2)
            x0_val = float(x0)
            k_lo = int(skellam.ppf(1e-7, m1_val, m2_val))
            k_hi = int(skellam.ppf(1 - 1e-7, m1_val, m2_val))
            ks = np.arange(k_lo, k_hi + 1)
            pmf_k = skellam.pmf(ks, m1_val, m2_val)
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
        """Return test parameters for Skellam."""
        params1 = {"mu1": 3.0, "mu2": 2.0}
        params2 = {"mu1": 5.0, "mu2": 1.0}
        return [params1, params2]
