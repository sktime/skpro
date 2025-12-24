# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Truncated Normal probability distribution."""

__author__ = ["ShreeshaM07"]

from scipy.stats import rv_continuous, truncnorm

from skpro.distributions.adapters.scipy import _ScipyAdapter


class TruncatedNormal(_ScipyAdapter):
    """A truncated normal probability distribution.

    Most methods wrap ``scipy.stats.truncnorm``.
    It truncates the normal distribution at
    the abscissa ``l_trunc`` and ``r_trunc``.

    Note: The truncation parameters passed
    is internally shifted to be centred at
    mean and scaled by sigma.

    Parameters
    ----------
    mu : float or array of float (1D or 2D)
        mean of the normal distribution
    sigma : float or array of float (1D or 2D), must be positive
        standard deviation of the normal distribution
    l_trunc : float or array of float (1D or 2D)
        Left truncation abscissa.
    r_trunc : float or array of float (1D or 2D)
        Right truncation abscissa.
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> from skpro.distributions.truncated_normal import TruncatedNormal

    >>> d = TruncatedNormal(\
            mu=[[0, 1], [2, 3], [4, 5]],\
            sigma= 1,\
            l_trunc= [[-0.1,0.5],[1.5,2.4],[4.1,5]],\
            r_trunc= [[0.8,2],[4,5],[5,7]]\
        )
    """

    _tags = {
        "capabilities:approx": ["energy", "pdfnorm"],
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, l_trunc, r_trunc, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        self.l_trunc = l_trunc
        self.r_trunc = r_trunc
        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self) -> rv_continuous:
        return truncnorm

    def _get_scipy_param(self):
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]
        l_trunc = self._bc_params["l_trunc"]
        r_trunc = self._bc_params["r_trunc"]
        # shift it to be centred at mu and sigma
        a = (l_trunc - mu) / sigma
        b = (r_trunc - mu) / sigma
        return [], {
            "loc": mu,
            "scale": sigma,
            "a": a,
            "b": b,
        }

    def _energy_self(self):
        """Energy of self, w.r.t. self
        (expected |X-Y| for i.i.d. X,Y ~ TruncatedNormal).
        """
        import numpy as np
        from scipy.integrate import quad
        from scipy.stats import truncnorm

        mu = np.asarray(self._bc_params["mu"])
        sigma = np.asarray(self._bc_params["sigma"])
        l_trunc = np.asarray(self._bc_params["l_trunc"])
        r_trunc = np.asarray(self._bc_params["r_trunc"])
        mu_b, sigma_b, l_b, r_b = np.broadcast_arrays(mu, sigma, l_trunc, r_trunc)
        result = np.empty_like(mu_b, dtype=float)
        it = np.nditer(
            [mu_b, sigma_b, l_b, r_b, result],
            flags=("multi_index",),
            op_flags=(
                ("readonly",),
                ("readonly",),
                ("readonly",),
                ("readonly",),
                ("writeonly",),
            ),
        )
        for m, s, l, r, out in it:
            a = (l.item() - m.item()) / s.item()
            b = (r.item() - m.item()) / s.item()

            a_val = a
            b_val = b
            m_val = m.item()
            s_val = s.item()

            def cdf(x, a_val=a_val, b_val=b_val, m_val=m_val, s_val=s_val):
                return truncnorm.cdf(x, a_val, b_val, loc=m_val, scale=s_val)

            def integrand(x, cdf=cdf):
                F = cdf(x)
                return 2 * F * (1 - F)
            val, _ = quad(integrand, l.item(), r.item(), limit=200)
            out[...] = val
        # Always flatten to 1D of length n_rows for DataFrame compatibility
        n_rows = 1 if self.index is None else len(self.index)
        result = np.asarray(result).reshape(-1)
        if result.shape[0] != n_rows:
            result = result.reshape(n_rows, -1).mean(axis=1)
        if self.index is None and n_rows == 1:
            return result.item()
        return result

    def _energy_x(self, x):
        """Energy of self, w.r.t. a constant frame x
        (expected |X-x| for X ~ TruncatedNormal).
        """
        import numpy as np
        from scipy.integrate import quad
        from scipy.stats import truncnorm

        mu = np.asarray(self._bc_params["mu"])
        sigma = np.asarray(self._bc_params["sigma"])
        l_trunc = np.asarray(self._bc_params["l_trunc"])
        r_trunc = np.asarray(self._bc_params["r_trunc"])
        x = np.asarray(x)
        mu_b, sigma_b, l_b, r_b, x_b = np.broadcast_arrays(
            mu, sigma, l_trunc, r_trunc, x
        )
        result = np.empty_like(mu_b, dtype=float)
        it = np.nditer(
            [mu_b, sigma_b, l_b, r_b, x_b, result],
            flags=("multi_index",),
            op_flags=(
                ("readonly",),
                ("readonly",),
                ("readonly",),
                ("readonly",),
                ("readonly",),
                ("writeonly",),
            ),
        )
        for m, s, l, r, x0, out in it:
            a = (l.item() - m.item()) / s.item()
            b = (r.item() - m.item()) / s.item()

            a_val = a
            b_val = b
            m_val = m.item()
            s_val = s.item()
            x0_val = x0.item()

            def integrand(
                t, a_val=a_val, b_val=b_val, m_val=m_val, s_val=s_val, x0_val=x0_val
            ):
                return np.abs(t - x0_val) * truncnorm.pdf(
                    t, a_val, b_val, loc=m_val, scale=s_val
                )
            val, _ = quad(integrand, l.item(), r.item(), limit=200)
            out[...] = val
        # Always flatten to 1D of length n_rows for DataFrame compatibility
        n_rows = 1 if self.index is None else len(self.index)
        result = np.asarray(result).reshape(-1)
        if result.shape[0] != n_rows:
            result = result.reshape(n_rows, -1).mean(axis=1)
        if self.index is None and n_rows == 1:
            return result.item()
        return result

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return {
            "mu": [[0, 1], [2, 3], [4, 5]],
            "sigma": 1,
            "l_trunc": [[-0.1, 0.5], [1.5, 2.4], [4.1, 5]],
            "r_trunc": [[0.8, 2], [4, 5], [5, 7]],
        }
