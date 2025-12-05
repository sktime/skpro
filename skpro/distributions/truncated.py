"""Truncated distributions."""
from typing import Tuple, Union

import numpy as np

from skpro.distributions.base import BaseDistribution


class TruncatedDistribution(BaseDistribution):
    r"""A truncated distribution.

    Given a univariate distribution, this distribution samples from the base
    distribution but truncates the values to lie between a specified lower and
    upper bound.
    Mathematically, it can be expressed as:

    .. math::
        Y \sim f(y \vert a \lt y \leq b) = \frac{f(y)}{F(b) - F(a)},

    where :math:`a` and :math:`b` is the lower and upper bound respectively, and
    :math:`f(y)` is the probability mass/density function.

    Parameters
    ----------
    distribution : BaseDistribution
        The distribution to truncate.

    lower : Union[float, int], optional
        The lower bound below which values are truncated.
        By default, this bound is exclusive (see ``inclusive_lower``).

    upper : Union[float, int], optional
        The upper bound above which values are truncated.

    inclusive_lower : bool, optional, default = False
        If True, the lower bound is inclusive (x >= lower).

    Examples
    --------
    >>> from skpro.distributions import Normal, TruncatedDistribution
    >>>
    >>> base = Normal(mu=1.0, sigma=1.0)
    >>> truncated = TruncatedDistribution(base, lower=0.0, upper=5.0)
    >>> samples = truncated.sample(1000)
    """

    _tags = {
        "capabilities:approx": ["energy", "mean", "var"],
        "capabilities:exact": [
            "ppf",
            "log_pmf",
            "log_pdf",
            "pmf",
            "pdf",
            "cdf",
        ],
        "distr:measuretype": "mixed",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(
        self,
        distribution: BaseDistribution,
        *,
        lower: Union[float, int] = None,
        upper: Union[float, int] = None,
        inclusive_lower: bool = False,
        index=None,
        columns=None,
    ):
        self.distribution = distribution
        self.lower = lower
        self.upper = upper
        self.inclusive_lower = inclusive_lower

        super().__init__(
            index=index if index is not None else distribution.index,
            columns=columns if columns is not None else distribution.columns,
        )

        # the capabilities of this (self) distribution are exact
        # if and only if the capabilities of the inner distribution are exact
        self_exact_capas = self.get_tag("capabilities:exact", []).copy()
        self_approx_capas = self.get_tag("capabilities:approx", []).copy()
        distr_exact_capas = distribution.get_tag("capabilities:exact", []).copy()
        for capa in self_exact_capas:
            if capa not in distr_exact_capas:
                self_exact_capas.remove(capa)
                self_approx_capas.append(capa)

        self.set_tags(**{"capabilities:exact": self_exact_capas})
        self.set_tags(**{"capabilities:approx": self_approx_capas})

        # the measure type of this distribution is discrete if the inner distribution
        # is discrete, and it is continuous if the inner distribution is continuous;
        # if the measure type of the inner distribution is mixed,
        # the result is indeterminate, but we keep it as mixed for API reasons
        inner_measuretype = distribution.get_tag("distr:measuretype", "mixed")
        self.set_tags(**{"distr:measuretype": inner_measuretype})

        inner_paramtype = distribution.get_tag("distr:paramtype", "parametric")
        if inner_paramtype != "parametric":
            self.set_tags(**{"distr:paramtype": inner_paramtype})

    def _get_low_high_prob(self) -> Tuple[float, float]:
        if self.lower is not None:
            prob_at_lower = self.distribution.cdf(self.lower)
            if self.inclusive_lower:
                measure_type = self.get_tag("distr:measuretype")

                # If continuous, P(X=lower) is 0, so CDF(lower) is already correct
                # If discrete, then P(X < lower) = CDF(lower) - P(X=lower)
                if measure_type == "discrete":
                    prob_mass_at_lower = self.distribution.pmf(self.lower)
                    prob_at_lower = prob_at_lower - prob_mass_at_lower
        else:
            prob_at_lower = 0.0

        prob_at_upper = (
            self.distribution.cdf(self.upper) if self.upper is not None else 1.0
        )

        return prob_at_lower, prob_at_upper

    def _calculate_density(self, x: np.ndarray, fun, as_log: bool):
        inf = float("inf")

        lower_bound = self.lower if self.lower is not None else -inf
        upper_bound = self.upper if self.upper is not None else inf

        if self.inclusive_lower:
            is_valid = (x >= lower_bound) & (x <= upper_bound)
        else:
            is_valid = (x > lower_bound) & (x <= upper_bound)

        prob_base = fun(x)
        cdf_lower, cdf_upper = self._get_low_high_prob()

        normalizer = cdf_upper - cdf_lower

        if as_log:
            prob_truncated = prob_base - np.log(normalizer)
        else:
            prob_truncated = prob_base / normalizer

        return np.where(is_valid, prob_truncated, -inf if as_log else 0.0)

    def _ppf(self, p):
        prob_at_lower, prob_at_upper = self._get_low_high_prob()
        factor = prob_at_upper - prob_at_lower
        offset = prob_at_lower

        shifted_p = p * factor + offset

        # NB: some parameterizations of discrete distributions may lead to rounding
        # errors, causing "off-by-one" issues in the quantile function. As a
        # workaround we subtract a small epsilon value.
        eps = np.finfo(p.dtype).eps
        shifted_p = np.clip(shifted_p - eps, 0.0, 1.0)

        return self.distribution.ppf(shifted_p)

    def _cdf(self, x):
        prob_at_lower, prob_at_upper = self._get_low_high_prob()

        return (self.distribution.cdf(x) - prob_at_lower) / (
            prob_at_upper - prob_at_lower
        )

    def _log_pdf(self, x):
        return self._calculate_density(x, self.distribution.log_pdf, as_log=True)

    def _pdf(self, x):
        return self._calculate_density(x, self.distribution.pdf, as_log=False)

    def _pmf(self, x):
        return self._calculate_density(x, self.distribution.pmf, as_log=False)

    def _log_pmf(self, x):
        return self._calculate_density(x, self.distribution.log_pmf, as_log=True)

    def _iloc(self, rowidx=None, colidx=None):
        distr = self.distribution.iloc[rowidx, colidx]

        if rowidx is not None:
            new_index = self.index[rowidx]
        else:
            new_index = self.index

        if colidx is not None:
            new_columns = self.columns[colidx]
        else:
            new_columns = self.columns

        cls = type(self)
        return cls(
            distribution=distr,
            lower=self.lower,
            upper=self.upper,
            inclusive_lower=self.inclusive_lower,
            index=new_index,
            columns=new_columns,
        )

    def _iat(self, rowidx=None, colidx=None):
        if rowidx is None or colidx is None:
            raise ValueError("iat method requires both row and column index")
        self_subset = self.iloc[[rowidx], [colidx]]

        return type(self)(
            distribution=self_subset.distribution.iat[0, 0],
            lower=self.lower,
            upper=self.upper,
            inclusive_lower=self.inclusive_lower,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):  # noqa: D102
        import pandas as pd

        from skpro.distributions import Normal, Poisson

        # scalar
        dist = Poisson(mu=1.0)
        params1 = {
            "distribution": dist,
            "lower": 0,
            "upper": 5,
        }

        # array
        idx = pd.Index([1, 2])
        cols = pd.Index(["a", "b"])
        n_array = Poisson(mu=[[1, 2], [3, 4]], columns=cols, index=idx)
        params2 = {
            "distribution": n_array,
            "lower": 0,
            "upper": 5,
            "index": idx,
            "columns": cols,
        }

        # scalar
        dist = Normal(mu=1.0, sigma=1.0)
        params3 = {
            "distribution": dist,
            "lower": 0,
            "upper": 5,
        }

        # array
        idx = pd.Index([1, 2])
        cols = pd.Index(["a", "b"])
        n_array = Normal(mu=[[1, 2], [3, 4]], sigma=1.0, columns=cols, index=idx)
        params4 = {
            "distribution": n_array,
            "lower": 0,
            "upper": 5,
            "index": idx,
            "columns": cols,
        }
        # inclusive lower test parameter
        params5 = {"distribution": dist, "lower": 0.0, "inclusive_lower": True}

        return [params1, params2, params3, params4, params5]
