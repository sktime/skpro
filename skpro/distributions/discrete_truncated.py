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

    def __init__(self, distribution: BaseDistribution, lower_bound: int):
        super().__init__()
        self.distribution = distribution
        self.lower_bound = lower_bound

    def _sample(self, n_samples: int):
        u = Uniform(0.0, 1.0).sample(n_samples)
        return self._ppf(u)

    def _log_pdf(self, x):
        is_invalid = x <= self.low

        log_prob_base = self.distribution.log_prob(x)

        log_prob_at_zero = self.base_dist.log_prob(self.low)
        log_normalizer = np.log1p(-np.exp(log_prob_at_zero))

        log_prob_truncated = log_prob_base - log_normalizer

        return np.where(is_invalid, -np.inf, log_prob_truncated)

    def _ppf(self, u):
        low_cdf = self.distribution.cdf(self.lower_bound)
        normalizer = 1.0 - low_cdf
        x = low_cdf + normalizer * u

        return self.distribution.ppf(x)