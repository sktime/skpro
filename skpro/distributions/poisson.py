# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Poisson probability distribution."""

__author__ = ["fkiraly", "malikrafsan"]

from scipy.stats import poisson, rv_discrete

from skpro.distributions.adapters.scipy import _ScipyAdapter


class Poisson(_ScipyAdapter):
    """Poisson distribution.

    Most methods wrap ``scipy.stats.poisson``.

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        mean of the distribution
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions import Poisson

    >>> distr = Poisson(mu=[[1, 1], [2, 3], [4, 5]])
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pmf", "log_pmf", "cdf", "ppf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, index=None, columns=None):
        self.mu = mu
        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_discrete:
        return poisson

    def _get_scipy_param(self) -> dict:
        mu = self._bc_params["mu"]
        return [mu], {}

    def _energy_self(self):
        """Energy of self, w.r.t. self (expected |X-Y| for i.i.d. X,Y ~ Poisson)."""
        import numpy as np
        from scipy.stats import poisson

        mu = np.atleast_1d(self._bc_params["mu"])
        shape = mu.shape
        result = np.zeros(shape)
        for idx in np.ndindex(shape):
            m = mu[idx]
            max_k = int(np.ceil(poisson.ppf(0.999, m)))
            pmf = poisson.pmf(np.arange(0, max_k + 1), m)
            energy = 0.0
            for i in range(max_k + 1):
                for j in range(max_k + 1):
                    energy += pmf[i] * pmf[j] * abs(i - j)
            result[idx] = energy
        # flatten to 1D if needed for DataFrame compatibility
        if result.ndim > 1:
            result = result.reshape(result.shape[0], -1).sum(axis=1)
        # scalarize if needed
        if getattr(self, "index", None) is None and result.size == 1:
            return result.item()
        return result

    def _energy_x(self, x):
        """Energy of self, w.r.t. a constant frame x.

        Expected |X-x| for X ~ Poisson.
        """
        import numpy as np
        from scipy.stats import poisson

        mu = np.atleast_1d(self._bc_params["mu"])
        x = np.atleast_1d(x)
        shape = np.broadcast(mu, x).shape
        result = np.zeros(shape)
        for idx in np.ndindex(shape):
            m = np.broadcast_to(mu, shape)[idx]
            x0 = np.broadcast_to(x, shape)[idx]
            max_k = int(np.ceil(poisson.ppf(0.999, m)))
            pmf = poisson.pmf(np.arange(0, max_k + 1), m)
            energy = 0.0
            for k in range(max_k + 1):
                energy += pmf[k] * abs(k - x0)
            result[idx] = energy
        # flatten to 1D if needed for DataFrame compatibility
        if result.ndim > 1:
            result = result.reshape(result.shape[0], -1).sum(axis=1)
        # scalarize if needed
        if getattr(self, "index", None) is None and result.size == 1:
            return result.item()
        return result

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {"mu": [[1, 1], [2, 3], [4, 5]]}
        params2 = {
            "mu": 0.1,
            "index": [1, 2, 5],
            "columns": ["a", "b"],
        }
        return [params1, params2]
