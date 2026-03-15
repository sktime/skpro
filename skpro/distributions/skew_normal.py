# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

"""Skew-Normal probability distribution."""

import numpy as np
from scipy.stats import skewnorm

from skpro.distributions.adapters.scipy._distribution import _ScipyAdapter


class SkewNormal(_ScipyAdapter):
    r"""Skew-Normal Probability Distribution.

    The skew-normal distribution generalizes the normal distribution by introducing
    a shape parameter :math:`\\alpha` to control skewness. It is parameterized by
    ``mu``, ``sigma``, and ``alpha``:

    .. math:: f(x; \alpha, \\mu, \\sigma) = \frac{2}{\\sigma}
          \\phi\\left(\frac{x - \\mu}{\\sigma}\right)
          \\Phi\\left(\alpha \frac{x - \\mu}{\\sigma}\right)

    where :math:`\\phi` and :math:`\\Phi` are the pdf and cdf of the standard normal
    distribution, respectively, :math:`\\mu` is the location parameter,
    :math:`\\sigma` is the scale parameter (must be positive), and
    :math:`\alpha` is the shape parameter controlling skewness.

    Parameters
    ----------
    mu : float
        Location parameter of the distribution (mean shift).
    sigma : float
        Scale parameter of the distribution (standard deviation), must be positive.
    alpha : float
        Skewness parameter of the distribution:
        - A positive value skews to the right.
        - A negative value skews to the left.
        - A value of zero results in a standard normal distribution.
    index : array-like, optional, default=None
        Index for the distribution, providing pandas-like indexing for rows.
    columns : array-like, optional, default=None
        Columns for the distribution, providing pandas-like indexing for columns.

    Examples
    --------
    >>> from skpro.distributions import SkewNormal
    >>> s = SkewNormal(mu=0, sigma=1, alpha=3)
    """

    _tags = {
        "authors": ["Spinachboul"],
        "python_dependencies": ["scipy"],
        "distr:measuretype": "continuous",
        "capabilities:approx": ["pdfnorm"],
        "capabilities:exact": ["mean", "var", "energy", "pdf", "log_pdf", "cdf", "ppf"],
        "broadcast_init": "on",
    }

    def __init__(self, mu, sigma, alpha=0, index=None, columns=None):
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.index = index
        self.columns = columns

        super().__init__(index=index, columns=columns)

    def _get_scipy_object(self):
        return skewnorm

    def _get_scipy_param(self):
        mu = self._bc_params["mu"]
        sigma = self._bc_params["sigma"]
        alpha = self._bc_params["alpha"]

        return [], {
            "loc": mu,
            "scale": sigma,
            "a": alpha,
        }

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        For SkewNormal(mu, sigma, alpha), :math:`\mathbb{E}|X-Y|` is computed via:

        .. math:: \mathbb{E}|X-Y| = 2 \int_{-\infty}^{\infty} F(t)(1-F(t))\,dt

        using numerical integration over the CDF.
        """
        from scipy.integrate import quad

        mu = np.asarray(self._bc_params["mu"])
        sigma = np.asarray(self._bc_params["sigma"])
        alpha = np.asarray(self._bc_params["alpha"])
        mu_b, sigma_b, alpha_b = np.broadcast_arrays(mu, sigma, alpha)
        result = np.empty_like(mu_b, dtype=float)

        it = np.nditer(
            [mu_b, sigma_b, alpha_b, result],
            flags=["multi_index"],
            op_flags=[["readonly"], ["readonly"], ["readonly"], ["writeonly"]],
        )
        for mm, ss, aa, out in it:
            mm_val = float(mm)
            ss_val = float(ss)
            aa_val = float(aa)

            def integrand(t, mm=mm_val, ss=ss_val, aa=aa_val):
                F = skewnorm.cdf(t, a=aa, loc=mm, scale=ss)
                return 2 * F * (1 - F)

            val, _ = quad(integrand, -np.inf, np.inf, limit=200)
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

        :math:`\mathbb{E}[|X-x|]` for X ~ SkewNormal(mu, sigma, alpha),
        computed via numerical integration.
        """
        from scipy.integrate import quad

        mu = np.asarray(self._bc_params["mu"])
        sigma = np.asarray(self._bc_params["sigma"])
        alpha = np.asarray(self._bc_params["alpha"])
        x_arr = np.asarray(x)
        mu_b, sigma_b, alpha_b = np.broadcast_arrays(mu, sigma, alpha)
        _, x_b = np.broadcast_arrays(mu_b, x_arr)
        result = np.empty_like(mu_b, dtype=float)

        it = np.nditer(
            [mu_b, sigma_b, alpha_b, x_b, result],
            flags=["multi_index"],
            op_flags=[
                ["readonly"],
                ["readonly"],
                ["readonly"],
                ["readonly"],
                ["writeonly"],
            ],
        )
        for mm, ss, aa, x0, out in it:
            mm_val = float(mm)
            ss_val = float(ss)
            aa_val = float(aa)
            x0_val = float(x0)

            def integrand(t, mm=mm_val, ss=ss_val, aa=aa_val, x0=x0_val):
                return abs(t - x0) * skewnorm.pdf(t, a=aa, loc=mm, scale=ss)

            val, _ = quad(integrand, -np.inf, np.inf, limit=200)
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
        r"""Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class. Each dict represents
            parameters to construct a test instance:
            - `MyClass(**params)` or `MyClass(**params[i])` creates a valid test.
            - `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        return [
            {"mu": -1, "sigma": 2},
            {"mu": 0, "sigma": 1, "alpha": -2},
        ]
