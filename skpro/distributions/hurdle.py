"""Hurdle distribution implementation."""

import numpy as np
from numpy.typing import ArrayLike

from skpro.distributions.base import BaseDistribution
from skpro.distributions.discrete_truncated import LeftTruncatedDiscrete


class Hurdle(BaseDistribution):
    """A Hurdle distribution.

    Combines a Bernoulli gate for zero vs. non-zero outcomes with a zero-truncated
    distribution for the positive outcomes.

    Parameters
    ----------
    p : np.ndarray
        The probability of getting a non-zero value.

    distribution : LeftTruncatedDiscrete
        The zero-truncated distribution for positive outcomes.

    """

    _tags = {
        "capabilities:approx": ["energy", "pmf", "cdf"],
        "capabilities:exact": ["ppf", "mean", "var", "log_pmf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(
        self,
        p: ArrayLike,
        distribution: LeftTruncatedDiscrete,
        index=None,
        columns=None,
    ):
        assert (
            distribution.lower_bound == 0
        ), "The positive distribution must be zero-truncated."

        self.p = p
        self.distribution = distribution

        super().__init__(index=index, columns=columns)

    def _log_pmf(self, x):
        log_prob_zero = -np.log1p(self.p)
        log_prob_gate_pass = -np.log1p(np.reciprocal(self.p))

        log_prob_positive_value = self.distribution.log_pmf(x)

        log_prob_positive = log_prob_gate_pass + log_prob_positive_value

        is_zero = x == 0
        return np.where(is_zero, log_prob_zero, log_prob_positive)

    def _mean(self):
        return self.p * self.distribution.mean()

    def _ppf(self, p):
        prob_zero = 1.0 - self.p

        q_rescaled = (p - prob_zero) / self.p

        q_rescaled = np.clip(q_rescaled, 0.0, 1.0)
        y_positive = self.distribution.ppf(q_rescaled)

        return np.where(p <= prob_zero, 0.0, y_positive)

    @classmethod
    def get_test_params(cls, parameter_set="default"):  # noqa: D102
        from skpro.distributions import LeftTruncatedDiscrete, Poisson

        left_truncated_discrete = LeftTruncatedDiscrete(
            Poisson(mu=1.0),
            lower_bound=0,
        )

        params_1 = {
            "p": 0.3,
            "distribution": left_truncated_discrete,
        }

        return [params_1]

    # TODO: this is duplicated now and will also be for `TransformedDistribution`,
    #  perhaps add a mixin for this functionality?
    def _iloc(self, rowidx=None, colidx=None):
        distr = self.distribution.iloc[rowidx, colidx]
        p = self.p

        if rowidx is not None:
            new_index = self.index[rowidx]

            if isinstance(self.p, np.ndarray) and self.p.ndim > 0:
                p = p[rowidx]
        else:
            new_index = self.index

        if colidx is not None:
            new_columns = self.columns[colidx]

            if isinstance(self.p, np.ndarray) and self.p.ndim > 0:
                p = p[:, colidx]
        else:
            new_columns = self.columns

        cls = type(self)
        return cls(
            p=p,
            distribution=distr,
            index=new_index,
            columns=new_columns,
        )

    def _iat(self, rowidx=None, colidx=None):
        raise NotImplementedError("")
