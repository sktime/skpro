"""Hurdle distribution implementation."""
import numpy as np
from numpy.typing import ArrayLike

from skpro.distributions.base import BaseDistribution
from skpro.distributions.left_truncated import LeftTruncated
from skpro.distributions.truncated import TruncatedDistribution


def _set_to_constant_where_negative(mask: np.ndarray, x: np.ndarray, const: float):
    return np.where(mask, const, x)


# TODO: how to handle index/columns in these transformed distributions? must they be
#  the same as the original distribution?
class Hurdle(BaseDistribution):
    r"""A Hurdle distribution.

    Combines a Bernoulli gate for zero vs. non-zero outcomes with a zero-truncated
    distribution for the positive outcomes. Mathematically this can be expressed as:

    .. math::
        Y_t = \begin{cases}
                X \sim f(x \vert x > 0) &\text{ with probability } \pi, \\
                0 &\text{ with probability } 1 - \pi,
              \end{cases}

    where :math:`\pi` is the probability of getting a non-zero value, and
    :math:`f(x \vert x > 0)` is the probability mass function of the zero-truncated
    distribution.

    Parameters
    ----------
    p : ArrayLike
        The probability of getting a non-zero value.

    distribution : BaseDistribution
        Arbitrary distribution.

    Examples
    --------
    >>> from skpro.distributions import Normal, Hurdle
    >>>
    >>> base = Normal(mu=1.0, sigma=1.0)
    >>> hurdle = Hurdle(0.5, base)
    >>> samples = hurdle.sample(1000)
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
        p: ArrayLike,
        distribution: BaseDistribution,
        index=None,
        columns=None,
    ):
        if isinstance(p, np.ndarray) and p.ndim == 1:
            raise ValueError("p must be a scalar or a 2D array.")
        elif isinstance(p, np.ndarray) and p.ndim == 2:
            assert (
                p.shape[0] == distribution.shape[0]
            ), "If p is a 2D array, it must match the shape of the distribution."

        self.p = p
        self.distribution = distribution

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

        # the measuretype of this distribution is discrete if the inner distribution
        # is discrete, otherwise it is mixed
        inner_measuretype = distribution.get_tag("distr:measuretype", "mixed")
        if inner_measuretype == "discrete":
            self.set_tags(**{"distr:measuretype": "discrete"})

        inner_paramtype = distribution.get_tag("distr:paramtype", "parametric")
        if inner_paramtype != "parametric":
            self.set_tags(**{"distr:paramtype": inner_paramtype})

    # NB: not sure how much we need to conform with sklearn, but according to their
    # docs we shouldn't modify the input variables:
    # https://scikit-learn.org/stable/developers/develop.html
    @property
    def _truncated_distribution(self) -> TruncatedDistribution:
        if (
            isinstance(self.distribution, TruncatedDistribution)
            and self.distribution.lower == 0
        ):
            return self.distribution

        return LeftTruncated(
            self.distribution,
            lower=0,
            index=self.index,
            columns=self.columns,
        )

    def _log_pmf(self, x):
        log_prob_zero = np.log(1.0 - self.p)
        log_prob_hurdle = np.log(self.p)

        log_prob_positive_value = self._truncated_distribution.log_pmf(x)

        log_prob_positive = log_prob_hurdle + log_prob_positive_value

        is_zero = x == 0
        result = np.where(is_zero, log_prob_zero, log_prob_positive)

        return _set_to_constant_where_negative(x < 0.0, result, -np.inf)

    def _pmf(self, x):
        prob_zero = 1.0 - self.p
        prob_hurdle = self.p

        prob_positive_value = self._truncated_distribution.pmf(x)

        prob_positive = prob_hurdle * prob_positive_value

        is_zero = x == 0
        result = np.where(is_zero, prob_zero, prob_positive)

        return _set_to_constant_where_negative(x < 0.0, result, 0.0)

    def _log_pdf(self, x):
        log_prob_zero = np.log(1.0 - self.p)
        log_prob_hurdle = np.log(self.p)

        log_prob_positive_value = self._truncated_distribution.log_pdf(x)

        log_prob_positive = log_prob_hurdle + log_prob_positive_value

        is_zero = x == 0
        result = np.where(is_zero, log_prob_zero, log_prob_positive)

        return _set_to_constant_where_negative(x < 0.0, result, -np.inf)

    def _pdf(self, x):
        prob_zero = 1.0 - self.p
        prob_hurdle = self.p

        prob_positive_value = self._truncated_distribution.pdf(x)

        prob_positive = prob_hurdle * prob_positive_value

        is_zero = x == 0
        result = np.where(is_zero, prob_zero, prob_positive)

        return _set_to_constant_where_negative(x < 0.0, result, 0.0)

    def _mean(self):
        return self.p * self._truncated_distribution.mean()

    def _var(self):
        mean_positive = self._truncated_distribution.mean()
        var_positive = self._truncated_distribution.var()

        return self.p * var_positive + mean_positive * self.p * (1.0 - self.p)

    def _ppf(self, p):
        prob_zero = 1.0 - self.p

        q_rescaled = (p - prob_zero) / self.p

        q_rescaled = np.clip(q_rescaled, 0.0, 1.0)
        y_positive = self._truncated_distribution.ppf(q_rescaled)

        return np.where(p <= prob_zero, 0.0, y_positive)

    def _cdf(self, x):
        is_zero = x == 0.0
        prob_positive = self._truncated_distribution.cdf(x)

        result = np.where(
            is_zero,
            1.0 - self.p,
            (1.0 - self.p) + self.p * prob_positive,
        )

        return _set_to_constant_where_negative(x < 0.0, result, 0.0)

    @classmethod
    def get_test_params(cls, parameter_set="default"):  # noqa: D102
        import pandas as pd

        from skpro.distributions import NegativeBinomial

        # scalar
        params_1 = {
            "p": 0.3,
            "distribution": NegativeBinomial(mu=1.0, alpha=1.0),
        }

        # array 1
        mu = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        idx = pd.Index([0, 1])
        cols = pd.Index(["a", "b", "c"])

        negbin = NegativeBinomial(mu=mu, alpha=1.0, columns=cols, index=idx)
        params_2 = {
            "p": 0.3,
            "distribution": negbin,
            "index": idx,
            "columns": cols,
        }

        # array 2
        params_3 = {
            "p": np.array([0.2, 0.3]).reshape(-1, 1),
            "distribution": negbin,
            "index": idx,
            "columns": cols,
        }

        params = [params_1, params_2, params_3]

        # continuous
        from skpro.distributions import Normal

        params_1 = {
            "p": 0.3,
            "distribution": Normal(mu=1.0, sigma=1.0),
        }

        params_2 = {
            "p": 0.3,
            "distribution": Normal(mu=mu, sigma=1.0, columns=cols, index=idx),
            "index": idx,
            "columns": cols,
        }

        return params + [params_1, params_2]

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

            if isinstance(self.p, np.ndarray) and self.p.shape[-1] > 1:
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
        if rowidx is None or colidx is None:
            raise ValueError("iat method requires both row and column index")

        subset_p = self._subset_param(
            val=self.p,
            rowidx=rowidx,
            colidx=colidx,
            coerce_scalar=True,
        )

        self_subset = self.iloc[[rowidx], [colidx]]
        return type(self)(distribution=self_subset.distribution.iat[0, 0], p=subset_p)
