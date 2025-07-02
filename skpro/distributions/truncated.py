"""Truncated distributions."""

import numpy as np

from skpro.distributions.base import BaseDistribution


class _TruncatedDistribution(BaseDistribution):
    r"""A truncated discrete distribution _not_ including the lower bound.

    Given a univariate distribution, this distribution samples from the base
    distribution but truncates the values to lie between a specified lower and
    upper bound.
    Mathematically, it can be expressed as:

    .. math::
        Y \sim f(y \vert a \le y \leq b) = \frac{f(y)}{F(b) - F(a)},

    where :math:`a` and :math:`b` is the lower and upper bound respectively, and
    :math:`f(y)` is the probability mass/density function.

    Parameters
    ----------
    distribution : BaseDistribution
        The base discrete distribution from which to sample.

    lower : int, optional
        The lower bound below which values are truncated.

    upper : int, optional
        The upper bound above which values are truncated.

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
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(
        self,
        distribution: BaseDistribution,
        *,
        lower: float = None,
        upper: float = None,
        index=None,
        columns=None,
    ):
        self.distribution = distribution
        self.lower = lower
        self.upper = upper

        super().__init__(index=index, columns=columns)

    def _get_normalizer(self, as_log: bool = False):
        prob_at_lower = (
            self.distribution.cdf(self.lower) if self.lower is not None else 0.0
        )
        prob_at_upper = (
            self.distribution.cdf(self.upper) if self.upper is not None else 1.0
        )

        normalizer = prob_at_upper - prob_at_lower

        return np.log(normalizer) if as_log else normalizer

    def _calculate_density(self, x: np.ndarray, fun, as_log: bool):
        inf = float("inf")

        is_valid = (x > (self.lower or -inf)) & (x <= (self.upper or inf))

        prob_base = fun(x)
        normalizer = self._get_normalizer(as_log=as_log)

        if as_log:
            prob_truncated = prob_base - normalizer
        else:
            prob_truncated = prob_base / normalizer

        return np.where(is_valid, prob_truncated, -inf if as_log else 0.0)

    def _ppf(self, p):
        prob_at_lower = (
            self.distribution.cdf(self.lower) if self.lower is not None else 0.0
        )
        prob_at_upper = (
            self.distribution.cdf(self.upper) if self.upper is not None else 1.0
        )

        factor = prob_at_upper - prob_at_lower
        offset = prob_at_lower

        shifted_p = p * factor + offset
        return self.distribution.ppf(shifted_p)

    def _cdf(self, x):
        prob_at_lower = (
            self.distribution.cdf(self.lower) if self.lower is not None else 0.0
        )
        prob_at_upper = (
            self.distribution.cdf(self.upper) if self.upper is not None else 1.0
        )

        return (self.distribution.cdf(x) - prob_at_lower) / (
            prob_at_upper - prob_at_lower
        )

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
        )


class ContinuousTruncatedDistribution(_TruncatedDistribution):
    """A truncated continuous distribution _not_ including the lower bound.

    See :class:`_TruncatedDistribution` for more details.
    """

    _tags = {
        "capabilities:approx": ["energy", "mean", "var"],
        "capabilities:exact": [
            "ppf",
            "log_pdf",
            "pdf",
            "cdf",
        ],
        "distr:measuretype": "continuous",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def _log_pdf(self, x):
        return self._calculate_density(x, self.distribution.log_pdf, as_log=True)

    def _pdf(self, x):
        return self._calculate_density(x, self.distribution.pdf, as_log=False)

    @classmethod
    def get_test_params(cls, parameter_set="default"):  # noqa: D102
        import pandas as pd

        from skpro.distributions import Normal

        # scalar
        dist = Normal(mu=1.0, sigma=1.0)
        params1 = {
            "distribution": dist,
            "lower": 0,
            "upper": 5,
        }

        # array
        idx = pd.Index([1, 2])
        cols = pd.Index(["a", "b"])
        n_array = Normal(mu=[[1, 2], [3, 4]], sigma=1.0, columns=cols, index=idx)
        params2 = {
            "distribution": n_array,
            "lower": 0,
            "upper": 5,
            "index": idx,
            "columns": cols,
        }

        return [params1, params2]


class DiscreteTruncatedDistribution(_TruncatedDistribution):
    """A truncated discrete distribution _not_ including the lower bound.

    See :class:`_TruncatedDistribution` for more details.
    """

    _tags = {
        "capabilities:approx": ["energy", "mean", "var"],
        "capabilities:exact": [
            "ppf",
            "log_pmf",
            "pmf",
            "cdf",
        ],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def _log_pmf(self, x):
        return self._calculate_density(x, self.distribution.log_pmf, as_log=True)

    def _pmf(self, x):
        return self._calculate_density(x, self.distribution.pmf, as_log=False)

    @classmethod
    def get_test_params(cls, parameter_set="default"):  # noqa: D102
        import pandas as pd

        from skpro.distributions import Poisson

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

        return [params1, params2]


def TruncatedDistribution(
    distribution: BaseDistribution,
    *,
    lower: float = None,
    upper: float = None,
    index=None,
    columns=None,
) -> _TruncatedDistribution:
    """Create a truncated distribution.

    Main interface for creating truncated distributions.

    Parameters
    ----------
    distribution : BaseDistribution
        The base discrete distribution from which to sample.

    lower : int, optional
        The lower bound below which values are truncated.

    upper : int, optional
        The upper bound above which values are truncated.
    """
    measure_type = distribution.get_tag("distr:measuretype")
    if measure_type == "continuous":
        return ContinuousTruncatedDistribution(
            distribution=distribution,
            lower=lower,
            upper=upper,
            index=index,
            columns=columns,
        )

    if measure_type == "discrete":
        return DiscreteTruncatedDistribution(
            distribution=distribution,
            lower=lower,
            upper=upper,
            index=index,
            columns=columns,
        )

    raise ValueError(f"Unknown measure type: {measure_type}!")
