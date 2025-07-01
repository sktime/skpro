"""Left Truncated Discrete Distribution."""

import numpy as np

from skpro.distributions.base import BaseDistribution


# TODO: WIP
class TruncatedDistribution(BaseDistribution):
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
        "capabilities:approx": ["energy", "pmf", "cdf"],
        "capabilities:exact": [
            "ppf",
            "mean",
            "var",
            "log_pmf",
            "log_pdf",
            "pmf",
            "pdf",
        ],
        "distr:measuretype": "discrete",
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
        is_invalid = self.lower < x <= self.upper

        prob_base = fun(x)
        normalizer = self._get_normalizer(as_log=as_log)

        if as_log:
            prob_truncated = prob_base - normalizer
        else:
            prob_truncated = prob_base / normalizer

        return np.where(is_invalid, -np.inf if as_log else 0.0, prob_truncated)

    def _log_pmf(self, x):
        return self._calculate_density(x, self.distribution.log_pmf, as_log=True)

    def _log_pdf(self, x):
        return self._calculate_density(x, self.distribution.log_pdf, as_log=True)

    def _pmf(self, x):
        return self._calculate_density(x, self.distribution.pmf, as_log=False)

    def _pdf(self, x):
        return self._calculate_density(x, self.distribution.pdf, as_log=False)

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

    @classmethod
    def get_test_params(cls, parameter_set="default"):  # noqa: D102
        import pandas as pd

        from skpro.distributions import Poisson

        # scalar
        poisson = Poisson(mu=1.0)
        params1 = {
            "distribution": poisson,
            "lower": 0,
        }

        # array
        idx = pd.Index([1, 2])
        cols = pd.Index(["a", "b"])
        n_array = Poisson(mu=[[1, 2], [3, 4]], columns=cols, index=idx)
        params2 = {
            "distribution": n_array,
            "lower": 0,
            "index": idx,
            "columns": cols,
        }

        return [params1, params2]

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
            lower_bound=self.lower_bound,
            index=new_index,
            columns=new_columns,
        )

    def _iat(self, rowidx=None, colidx=None):
        if rowidx is None or colidx is None:
            raise ValueError("iat method requires both row and column index")
        self_subset = self.iloc[[rowidx], [colidx]]

        return type(self)(
            distribution=self_subset.distribution.iat[0, 0],
            lower_bound=self.lower_bound,
        )
