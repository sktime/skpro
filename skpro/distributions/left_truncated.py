"""Left Truncated Discrete Distribution."""

from skpro.distributions.base import BaseDistribution
from skpro.distributions.truncated import TruncatedDistribution


class LeftTruncated(TruncatedDistribution):
    r"""A left truncated distribution _not_ including the lower bound.

    See :class:`TruncatedDistribution` for more details.

    Parameters
    ----------
    distribution : BaseDistribution
        The base discrete distribution from which to sample.

    lower : int
        The lower bound below which values are truncated (excluded from sampling).

    """

    _tags = {
        "capabilities:approx": ["energy", "pmf", "cdf"],
        "capabilities:exact": ["ppf", "mean", "var", "log_pmf"],
        "distr:measuretype": "mixed",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(
        self, distribution: BaseDistribution, lower: float, index=None, columns=None
    ):
        super().__init__(
            distribution, lower=lower, upper=None, index=index, columns=columns
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
            index=new_index,
            columns=new_columns,
        )

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
