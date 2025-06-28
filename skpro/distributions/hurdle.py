import numpy as np

from skpro.distributions import Binomial
from skpro.distributions.base import BaseDistribution
from skpro.distributions.discrete_truncated import LeftTruncatedDiscrete


class Hurdle(BaseDistribution):
    """A Hurdle distribution.

    Combines a Bernoulli gate for zero vs. non-zero outcomes with a zero-truncated
    distribution for the positive outcomes.

    Parameters
    ----------
    probs : np.ndarray
        The probabilities of the Bernoulli gate (zero vs. positive).

    positive_dist : LeftTruncatedDiscrete
        The zero-truncated distribution for positive outcomes.
    """

    _tags = {
        "capabilities:approx": ["energy", "pmf", "cdf"],
        "capabilities:exact": ["ppf", "mean", "var", "log_pmf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, probs: np.ndarray, positive_dist: LeftTruncatedDiscrete, index=None, columns=None):
        super().__init__(index=index, columns=columns)
        assert positive_dist.lower_bound == 0, "The positive distribution must be zero-truncated."

        self.probs = probs
        self.positive_dist = positive_dist


    def sample(self, n_samples=None):
        is_positive = Binomial(n=1, p=self.probs).sample()
        positive_values = self.positive_dist.sample(n_samples)

        return np.where(is_positive, positive_values, 0.0)

    def log_pmf(self, value):
        log_prob_zero = -np.log1p(self.probs)
        log_prob_gate_pass = -np.log1p(np.reciprocal(self.probs))  # log(sigmoid(logits))

        log_prob_positive_value = self.positive_dist.log_pmf(value)

        log_prob_positive = log_prob_gate_pass + log_prob_positive_value

        is_zero = (value == 0)
        return np.where(is_zero, log_prob_zero, log_prob_positive)

    @property
    def mean(self):
        return self.probs * self.positive_dist.mean

    def ppf(self, x):
        prob_zero = 1.0 - self.probs

        q_rescaled = (x - prob_zero) / self.probs

        q_rescaled = np.clip(q_rescaled, 0.0, 1.0)
        y_positive = self.positive_dist.ppf(q_rescaled)

        return np.where(x <= prob_zero, 0.0, y_positive)
