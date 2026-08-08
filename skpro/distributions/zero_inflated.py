# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Zero-Inflated distribution compositor."""

__author__ = ["marrov"]

import numpy as np
from numpy.typing import ArrayLike

from skpro.distributions.base import BaseDistribution


def _set_to_constant_where_negative(mask: np.ndarray, x: np.ndarray, const: float):
    """Set values to constant where mask is True (for negative values)."""
    return np.where(mask, const, x)


class ZeroInflated(BaseDistribution):
    r"""A Zero-Inflated distribution compositor.

    Creates a zero-inflated version of any base distribution by mixing it with
    a point mass at zero. Unlike the Hurdle distribution, zeros can come from
    **both** the zero component **and** the base distribution.

    **Important:** This compositor is designed for distributions with non-negative
    support (e.g., Poisson, NegativeBinomial, Gamma). Using it with distributions
    that have negative support (e.g., Normal, Laplace) will produce incorrect
    probabilities for negative values, as they are clamped to zero.

    Mathematically, for a zero-inflated distribution with excess zero probability
    :math:`\pi` and base distribution :math:`f(x)`:

    .. math::
        P(X = x) = \begin{cases}
            \pi + (1-\pi) \cdot f(0) & \text{if } x = 0 \\
            (1-\pi) \cdot f(x) & \text{if } x > 0 \\
            0 & \text{if } x < 0
        \end{cases}

    The CDF is given by:

    .. math::
        F(x) = \begin{cases}
            0 & \text{if } x < 0 \\
            \pi + (1-\pi) \cdot F_{base}(x) & \text{if } x \geq 0
        \end{cases}

    The mean and variance are:

    .. math::
        E[X] = (1-\pi) \cdot E_{base}[X]

    .. math::
        Var(X) = (1-\pi) \cdot Var_{base}(X) + (1-\pi) \cdot \pi \cdot (E_{base}[X])^2

    Parameters
    ----------
    pi : ArrayLike
        The probability of excess zeros (zero-inflation probability).
        Must be in [0, 1).
    distribution : BaseDistribution
        The base distribution to be zero-inflated. Should have non-negative support.
    index : pd.Index, optional
        Index for the distribution.
    columns : pd.Index, optional
        Columns for the distribution.

    Examples
    --------
    >>> from skpro.distributions import Poisson, ZeroInflated
    >>>
    >>> base = Poisson(mu=2.0)
    >>> zi_poisson = ZeroInflated(pi=0.3, distribution=base)
    >>> samples = zi_poisson.sample(1000)

    See Also
    --------
    Hurdle : Hurdle distribution (zeros only from zero component, base truncated)
    ZINB : Zero-Inflated Negative Binomial (specialized implementation)
    ZIPoisson : Zero-Inflated Poisson (specialized implementation)
    """

    _tags = {
        "capabilities:approx": ["energy"],
        "capabilities:exact": ["ppf", "mean", "var", "log_pmf", "pmf", "cdf"],
        "distr:measuretype": "mixed",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(
        self,
        pi: ArrayLike,
        distribution: BaseDistribution,
        index=None,
        columns=None,
    ):
        # Validate shape and value constraints for pi
        if isinstance(pi, np.ndarray) and pi.ndim == 1:
            raise ValueError("pi must be a scalar or a 2D array.")
        elif isinstance(pi, np.ndarray) and pi.ndim == 2:
            if pi.shape[0] != distribution.shape[0]:
                raise ValueError(
                    "If pi is a 2D array, its first dimension must match "
                    "the distribution's shape[0]."
                )
            if pi.shape[1] not in {1, distribution.shape[1]}:
                raise ValueError(
                    "If pi is a 2D array, its second dimension must be 1 or match "
                    "the distribution's shape[1]."
                )

        # Validate pi values
        pi_arr = np.asarray(pi)
        if np.any(pi_arr < 0) or np.any(pi_arr >= 1):
            raise ValueError("pi must be in [0, 1)")

        # Check base distribution has non-negative support
        inner_measuretype = distribution.get_tag("distr:measuretype", "mixed")
        if inner_measuretype not in {"discrete", "continuous"}:
            # For mixed distributions, we can't easily check support
            pass
        else:
            # For safety, document that negative support is not handled
            # Users should not use ZeroInflated with distributions like Normal
            # unless they know the data is non-negative
            pass

        self.pi = pi
        self.distribution = distribution

        super().__init__(
            index=index if index is not None else distribution.index,
            columns=columns if columns is not None else distribution.columns,
        )

        # Update capabilities based on inner distribution
        self_exact_capas = self.get_tag("capabilities:exact", []).copy()
        self_approx_capas = self.get_tag("capabilities:approx", []).copy()
        distr_exact_capas = distribution.get_tag("capabilities:exact", []).copy()

        for capa in list(self_exact_capas):
            if capa not in distr_exact_capas:
                self_exact_capas.remove(capa)
                self_approx_capas.append(capa)

        self.set_tags(**{"capabilities:exact": self_exact_capas})
        self.set_tags(**{"capabilities:approx": self_approx_capas})

        # Set measuretype based on inner distribution
        inner_measuretype = distribution.get_tag("distr:measuretype", "mixed")
        if inner_measuretype == "discrete":
            self.set_tags(**{"distr:measuretype": "discrete"})

        inner_paramtype = distribution.get_tag("distr:paramtype", "parametric")
        if inner_paramtype != "parametric":
            self.set_tags(**{"distr:paramtype": inner_paramtype})

    def _log_pmf(self, x):
        """Return log probability mass function evaluated at x."""
        pi = self.pi
        base_pmf_at_0 = self.distribution.pmf(np.zeros_like(x))
        base_log_pmf = self.distribution.log_pmf(x)

        # For x=0: log(pi + (1-pi)*P_base(0))
        log_prob_zero = np.log(pi + (1 - pi) * base_pmf_at_0)
        # For x>0: log(1-pi) + log(P_base(x))
        log_prob_positive = np.log(1 - pi) + base_log_pmf

        is_zero = x == 0
        result = np.where(is_zero, log_prob_zero, log_prob_positive)

        return _set_to_constant_where_negative(x < 0.0, result, -np.inf)

    def _pmf(self, x):
        """Return probability mass function evaluated at x."""
        pi = self.pi
        base_pmf = self.distribution.pmf(x)
        base_pmf_at_0 = self.distribution.pmf(np.zeros_like(x))

        is_zero = x == 0
        result = np.where(
            is_zero,
            pi + (1 - pi) * base_pmf_at_0,
            (1 - pi) * base_pmf,
        )

        return _set_to_constant_where_negative(x < 0.0, result, 0.0)

    def _log_pdf(self, x):
        """Return log probability density function evaluated at x."""
        pi = self.pi
        base_pdf_at_0 = self.distribution.pdf(np.zeros_like(x))
        base_log_pdf = self.distribution.log_pdf(x)

        # For x=0: log(pi + (1-pi)*f_base(0))
        log_prob_zero = np.log(pi + (1 - pi) * base_pdf_at_0)
        # For x>0: log(1-pi) + log(f_base(x))
        log_prob_positive = np.log(1 - pi) + base_log_pdf

        is_zero = x == 0
        result = np.where(is_zero, log_prob_zero, log_prob_positive)

        return _set_to_constant_where_negative(x < 0.0, result, -np.inf)

    def _pdf(self, x):
        """Return probability density function evaluated at x."""
        pi = self.pi
        base_pdf = self.distribution.pdf(x)
        base_pdf_at_0 = self.distribution.pdf(np.zeros_like(x))

        is_zero = x == 0
        result = np.where(
            is_zero,
            pi + (1 - pi) * base_pdf_at_0,
            (1 - pi) * base_pdf,
        )

        return _set_to_constant_where_negative(x < 0.0, result, 0.0)

    def _mean(self):
        """Return mean of the distribution."""
        pi = self.pi
        base_mean = self.distribution.mean()
        return (1 - pi) * base_mean

    def _var(self):
        """Return variance of the distribution."""
        pi = self.pi
        base_mean = self.distribution.mean()
        base_var = self.distribution.var()

        # Var(X) = (1-pi) * Var_base + (1-pi) * pi * (E_base)^2
        return (1 - pi) * base_var + (1 - pi) * pi * base_mean**2

    def _ppf(self, p):
        """Return percent point function (inverse CDF) evaluated at p."""
        pi = self.pi
        base_cdf_at_0 = self.distribution.cdf(np.zeros_like(p))

        # Total probability at 0
        prob_zero = pi + (1 - pi) * base_cdf_at_0

        # For p <= prob_zero, return 0
        # For p > prob_zero, invert the base CDF
        # p = pi + (1-pi) * F_base(x)
        # => F_base(x) = (p - pi) / (1 - pi)
        q_rescaled = (p - pi) / (1 - pi)
        q_rescaled = np.clip(q_rescaled, 0.0, 1.0)

        # Subtract tiny epsilon to avoid floating-point precision issues where
        # q_rescaled ends up slightly larger than CDF(k) due to arithmetic,
        # which would cause ppf to return k+1 instead of k for discrete distributions
        eps = np.finfo(float).eps * 10
        q_rescaled = np.maximum(q_rescaled - eps, 0.0)

        y_positive = self.distribution.ppf(q_rescaled)

        return np.where(p <= prob_zero, 0.0, y_positive)

    def _cdf(self, x):
        """Return cumulative distribution function evaluated at x."""
        pi = self.pi
        base_cdf = self.distribution.cdf(x)

        # CDF: pi + (1-pi) * F_base(x) for x >= 0
        result = pi + (1 - pi) * base_cdf

        return _set_to_constant_where_negative(x < 0.0, result, 0.0)

    @classmethod
    def get_test_params(cls, parameter_set="default"):  # noqa: D102
        """Return testing parameter settings for the estimator."""
        import pandas as pd

        from skpro.distributions import NegativeBinomial, Poisson

        # scalar
        params_1 = {
            "pi": 0.3,
            "distribution": Poisson(mu=2.0),
        }

        # array with NegativeBinomial
        mu = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx = pd.Index([0, 1])
        cols = pd.Index(["a", "b", "c"])

        negbin = NegativeBinomial(mu=mu, alpha=1.0, columns=cols, index=idx)
        params_2 = {
            "pi": 0.3,
            "distribution": negbin,
            "index": idx,
            "columns": cols,
        }

        # array pi with NegativeBinomial
        params_3 = {
            "pi": np.array([[0.2], [0.3]]),
            "distribution": negbin,
            "index": idx,
            "columns": cols,
        }

        # Poisson with index
        pois = Poisson(mu=mu, columns=cols, index=idx)
        params_4 = {
            "pi": 0.3,
            "distribution": pois,
            "index": idx,
            "columns": cols,
        }

        return [params_1, params_2, params_3, params_4]

    def _iloc(self, rowidx=None, colidx=None):
        """Return subset of distribution by row and column indices."""
        distr = self.distribution.iloc[rowidx, colidx]
        pi = self.pi

        if rowidx is not None:
            new_index = self.index[rowidx]
            if isinstance(self.pi, np.ndarray) and self.pi.ndim > 0:
                pi = pi[rowidx]
        else:
            new_index = self.index

        if colidx is not None:
            new_columns = self.columns[colidx]
            if isinstance(self.pi, np.ndarray) and self.pi.shape[-1] > 1:
                pi = pi[:, colidx]
        else:
            new_columns = self.columns

        cls = type(self)
        return cls(
            pi=pi,
            distribution=distr,
            index=new_index,
            columns=new_columns,
        )

    def _iat(self, rowidx=None, colidx=None):
        """Return single element of distribution."""
        if rowidx is None or colidx is None:
            raise ValueError("iat method requires both row and column index")

        subset_pi = self._subset_param(
            val=self.pi,
            rowidx=rowidx,
            colidx=colidx,
            coerce_scalar=True,
        )

        self_subset = self.iloc[[rowidx], [colidx]]
        return type(self)(distribution=self_subset.distribution.iat[0, 0], pi=subset_pi)
