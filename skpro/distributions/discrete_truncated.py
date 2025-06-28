import numpy as np

from skpro.distributions import Uniform
from skpro.distributions.base import BaseDistribution


# TODO: Implement a general discrete truncation and let this inherit from it
class LeftTruncatedDiscrete(BaseDistribution):
    """A left truncated discrete distribution _not_ including the lower bound.

    This distribution samples from a given discrete distribution but excludes the
    values below a specified lower bound.

    Parameters
    ----------
    distribution : BaseDistribution
        The base discrete distribution from which to sample.

    lower_bound : int
        The lower bound below which values are truncated (excluded from sampling).
    """

    _tags = {
        "capabilities:approx": ["energy", "pmf", "cdf"],
        "capabilities:exact": ["ppf", "mean", "var", "log_pmf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, distribution: BaseDistribution, lower_bound: int, index=None, columns=None):
        assert distribution._tags["distr:measuretype"] == "discrete", ""

        self.distribution = distribution
        self.lower_bound = lower_bound

        super().__init__(index=index, columns=columns)

    def _sample(self, n_samples: int):
        u = Uniform(0.0, 1.0, index=self.index, columns=self.columns).sample(n_samples)
        return self._ppf(u)

    def _log_pmf(self, x):
        is_invalid = x <= self.lower_bound

        log_prob_base = self.distribution.log_pmf(x)

        log_prob_at_zero = self.distribution.log_pmf(self.lower_bound)
        log_normalizer = np.log1p(-np.exp(log_prob_at_zero))

        log_prob_truncated = log_prob_base - log_normalizer

        return np.where(is_invalid, -np.inf, log_prob_truncated)

    def _ppf(self, p):
        low_cdf = self.distribution.cdf(self.lower_bound)
        normalizer = 1.0 - low_cdf
        x = low_cdf + normalizer * p

        return self.distribution.ppf(x)

    def _mean(self):
        low_cdf = self.distribution.cdf(self.lower_bound)
        normalizer = 1.0 - low_cdf
        return self.distribution.mean() / normalizer

    def _var(self):
        low_cdf = self.distribution.cdf(self.lower_bound)
        normalizer = 1.0 - low_cdf
        mean = self._mean()
        return (self.distribution.var() + mean**2) / normalizer - mean**2

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        from skpro.distributions import Poisson

        poisson = Poisson(mu=1.0)
        params1 = {
            "distribution": poisson,
            "lower_bound": 0,
        }

        return [params1]
