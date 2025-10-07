"""Left Truncated Discrete Distribution."""
from typing import Union

from skpro.distributions.base import BaseDistribution
from skpro.distributions.truncated import TruncatedDistribution


class LeftTruncated(TruncatedDistribution):
    r"""A left truncated distribution _not_ including the lower bound.

    See :class:`TruncatedDistribution` for more details.

    Parameters
    ----------
    distribution : BaseDistribution
        The distribution to truncate from the left, _not_ including the lower bound.

    lower : int
        The lower bound below which values are truncated (excluded from sampling).

    """

    def __init__(
        self,
        distribution: BaseDistribution,
        lower: Union[float, int],
        index=None,
        columns=None,
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

        from skpro.distributions import NegativeBinomial

        # scalar
        dist = NegativeBinomial(mu=1.0, alpha=1.0)
        params1 = {
            "distribution": dist,
            "lower": 0,
        }

        # array
        idx = pd.Index([1, 2])
        cols = pd.Index(["a", "b"])
        n_array = NegativeBinomial(
            mu=[[1, 2], [3, 4]], alpha=1.0, columns=cols, index=idx
        )
        params2 = {
            "distribution": n_array,
            "lower": 0,
            "index": idx,
            "columns": cols,
        }

        return [params1, params2]
