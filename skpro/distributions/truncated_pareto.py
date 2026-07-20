"""Truncated Pareto probability distribution for skpro."""

import numpy as np
from scipy.stats import pareto

from skpro.distributions.base import BaseDistribution


class TruncatedPareto(BaseDistribution):
    r"""Truncated Pareto probability distribution.

    The truncated Pareto distribution is a continuous probability distribution that
    is a Pareto distribution restricted to the interval $[l, u]$. It is parameterized
    by a shape parameter $b > 0$, scale parameter $s > 0$, lower bound $l$, and
    upper bound $u$. Its probability density function (PDF) is:

    .. math::
        f(x; b, s, l, u) = \frac{b s^b x^{-(b+1)}}{F(u) - F(l)},
        \quad l \leq x \leq u

    where $F(x)$ is the CDF of the standard Pareto distribution.

    Parameters
    ----------
    b : float
        Shape parameter
    scale : float
        Scale parameter
    lower : float
        Lower truncation bound
    upper : float
        Upper truncation bound
    """

    _tags = {
        "authors": ["arnavk23"],
        "distr:measuretype": "continuous",
        "capabilities:exact": ["mean", "var", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, b, scale=1.0, lower=1.0, upper=10.0, index=None, columns=None):
        self.b = b
        self.scale = scale
        self.lower = lower
        self.upper = upper
        super().__init__(index=index, columns=columns)

    def _raw_moment(self, k):
        r"""Return the ``k``-th raw moment :math:`\mathbb{E}[X^k]`, entry-wise.

        With shape ``b``, scale ``s`` and truncation ``[l, u]``,

        .. math::
            \mathbb{E}[X^k] = \frac{b s^b}{Z} \int_l^u x^{k-b-1} dx,
            \quad Z = (s/l)^b - (s/u)^b.

        The integral equals :math:`(u^{k-b} - l^{k-b}) / (k-b)` for
        :math:`b \neq k`, and :math:`\ln(u/l)` at the removable singularity
        :math:`b = k`.
        """
        b = self._bc_params["b"]
        scale = self._bc_params["scale"]
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]

        norm = (scale / lower) ** b - (scale / upper) ** b
        exponent = k - b
        # k - b == 0 is a removable singularity; avoid 0/0 in the unused branch
        safe_exponent = np.where(exponent == 0, 1.0, exponent)
        integral = np.where(
            exponent == 0,
            np.log(upper / lower),
            (upper**exponent - lower**exponent) / safe_exponent,
        )
        return b * scale**b / norm * integral

    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        return self._raw_moment(1)

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        mean = self._raw_moment(1)
        return self._raw_moment(2) - mean**2

    def _pdf(self, x):
        b = self._bc_params["b"]
        scale = self._bc_params["scale"]
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]
        norm = pareto.cdf(upper, b, scale=scale) - pareto.cdf(lower, b, scale=scale)
        pdf = np.where(
            (x >= lower) & (x <= upper), pareto.pdf(x, b, scale=scale) / norm, 0.0
        )
        return pdf

    def _cdf(self, x):
        b = self._bc_params["b"]
        scale = self._bc_params["scale"]
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]
        norm = pareto.cdf(upper, b, scale=scale) - pareto.cdf(lower, b, scale=scale)
        cdf = (pareto.cdf(x, b, scale=scale) - pareto.cdf(lower, b, scale=scale)) / norm
        cdf = np.where(x < lower, 0.0, np.where(x > upper, 1.0, cdf))
        return cdf

    def _ppf(self, p):
        b = self._bc_params["b"]
        scale = self._bc_params["scale"]
        lower = self._bc_params["lower"]
        upper = self._bc_params["upper"]
        norm = pareto.cdf(upper, b, scale=scale) - pareto.cdf(lower, b, scale=scale)
        p_adj = p * norm + pareto.cdf(lower, b, scale=scale)
        return pareto.ppf(p_adj, b, scale=scale)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return test parameters for TruncatedPareto."""
        params1 = {"b": 2.0, "scale": 1.0, "lower": 1.0, "upper": 10.0}
        params2 = {"b": 3.0, "scale": 2.0, "lower": 2.0, "upper": 20.0}
        return [params1, params2]
