"""Left Truncated Discrete Distribution."""

from typing import Union

import pandas as pd

from skpro.distributions.base import BaseDistribution
from skpro.distributions.base._base import is_scalar_notnone
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

    Examples
    --------
    >>> from skpro.distributions import LeftTruncated, Poisson
    >>>
    >>> d = LeftTruncated(Poisson(mu=2), lower=0)
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
        if is_scalar_notnone(rowidx) and is_scalar_notnone(colidx):
            return self._iat(rowidx, colidx)
        if is_scalar_notnone(rowidx):
            rowidx = pd.Index([rowidx])
        if is_scalar_notnone(colidx):
            colidx = pd.Index([colidx])

        distr = self.distribution.iloc[rowidx, colidx]
        lower = self._subset_param(self.lower, rowidx=rowidx, colidx=colidx)

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
            lower=lower,
            index=new_index,
            columns=new_columns,
        )

    def _iat(self, rowidx=None, colidx=None):
        if rowidx is None or colidx is None:
            raise ValueError("iat method requires both row and column index")

        if self.ndim == 0:
            if rowidx != 0 or colidx != 0:
                raise IndexError(
                    f"index {rowidx}, {colidx} is out of bounds for scalar distribution"
                )
            distr_iat = self.distribution.iat[0, 0]
        else:
            self_subset = self.iloc[[rowidx], [colidx]]
            distr_iat = self_subset.distribution.iat[0, 0]

        lower_iat = self._subset_param(
            self.lower, rowidx=rowidx, colidx=colidx, coerce_scalar=True
        )

        return type(self)(
            distribution=distr_iat,
            lower=lower_iat,
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
