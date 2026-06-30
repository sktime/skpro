# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Histogram quantile parametrized distribution."""

__author__ = ["siddharth7113"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution, _DelegatedDistribution
from skpro.distributions.delta import Delta
from skpro.distributions.histogram import Histogram
from skpro.distributions.meanscale import MeanScale
from skpro.distributions.mixture import Mixture
from skpro.distributions.truncated import TruncatedDistribution


class HistogramQPD(_DelegatedDistribution):
    r"""Histogram quantile parametrized distribution.

    This distribution is parameterized by a set of quantiles and quantile points,
    quantiles :math:`q_1 < q_2 < \dots < q_N`
    at quantile points :math:`p_1, p_2, \dots, p_N`,
    with :math:`0 \le p_1 < p_2 < \dots < p_N \le 1`.

    Between consecutive quantiles the distribution is uniform, i.e., it has
    piecewise-constant density (a histogram) and piecewise-linear quantile function.
    On the interval :math:`[q_i, q_{i+1}]` the density is
    :math:`(p_{i+1} - p_i) / (q_{i+1} - q_i)`.

    This is the quantile-parameterized counterpart of ``Histogram`` (which is
    parameterized by bins and bin masses), in the same way that ``QPD_Empirical``
    is the quantile-parameterized counterpart of ``Empirical``. Unlike
    ``QPD_Empirical``, which has a piecewise-constant (sum-of-diracs) quantile
    function, ``HistogramQPD`` interpolates *linearly* between the quantile points.

    The behaviour below :math:`p_1` and above :math:`p_N` is controlled by the
    ``tails`` parameter, which decides how the remaining tail mass :math:`p_1`
    (lower) and :math:`1 - p_N` (upper) is allocated:

    * ``"uniform"`` (default): the tails continue at the density of the adjacent
      bin until the tail mass is used up. The support is bounded, the quantiles at
      the quantile points are matched exactly, and the distribution is continuous,
      so both CRPS and log-loss are valid metrics.
    * ``"none"``: no tail mass and no tail density. The interior bin masses are
      renormalized to sum to one, so the quantiles do not exactly match the
      parameterization.
    * ``"mass"``: dirac point masses of weight :math:`p_1` at :math:`q_1` and
      :math:`1 - p_N` at :math:`q_N`. The quantiles are matched exactly, but the
      distribution is mixed (continuous interior plus two atoms).
    * an ``skpro`` distribution ``d``: ``d`` is cut in half at zero, and the two
      halves are attached to the lower and upper tail (shifted to :math:`q_1` and
      :math:`q_N` respectively), carrying mass :math:`p_1` and :math:`1 - p_N`.

    Parameters
    ----------
    quantiles : pd.DataFrame with pd.MultiIndex
        quantile points
        first (lowest) index must be float, corresponding to the quantile point
        further indices are sample (instance) indices
        columns correspond to the variables
    tails : str or BaseDistribution, optional, default="uniform"
        tail behaviour, one of ``"uniform"``, ``"none"``, ``"mass"``,
        or an ``skpro`` ``BaseDistribution`` instance, see above
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Examples
    --------
    >>> import pandas as pd
    >>> from skpro.distributions import HistogramQPD

    >>> quantile_idx = pd.MultiIndex.from_product(
    ...     [[0.1, 0.5, 0.9], [0, 1, 2]], names=["quantile", "sample"]
    ... )
    >>> quantiles = pd.DataFrame(
    ...     [[0, 1], [2, 3], [4, 5], [1, 2], [4, 5], [7, 8], [2, 3], [6, 7], [10, 11]],
    ...     index=quantile_idx,
    ...     columns=["a", "b"],
    ... )
    >>> dist = HistogramQPD(quantiles, tails="uniform")
    >>> median = dist.ppf(0.5)
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["siddharth7113"],
        # estimator tags
        # --------------
        "capabilities:approx": ["pdfnorm", "energy"],
        "capabilities:exact": ["mean", "var", "cdf", "ppf", "pdf"],
        "distr:measuretype": "continuous",
    }

    def __init__(self, quantiles, tails="uniform", index=None, columns=None):
        self.quantiles = quantiles
        self.tails = tails
        self.index = index
        self.columns = columns

        p, q_all, samples, col_index = self._parse_quantiles(quantiles)

        row_labels = samples if index is None else index
        col_labels = col_index if columns is None else columns

        # select values by label position (duplicate- and order-safe)
        rows = samples.get_indexer(row_labels)
        cols = col_index.get_indexer(col_labels)
        q = q_all[rows][:, cols, :]

        self.delegate_ = self._build_delegate(p, q, tails, row_labels, col_labels)

        super().__init__(index=row_labels, columns=col_labels)

        # tag depends on the tail mode, so copy it from the delegate
        for tag in ("distr:measuretype", "capabilities:exact", "capabilities:approx"):
            self.set_tags(**{tag: self.delegate_.get_tag(tag)})

    def _parse_quantiles(self, quantiles):
        """Parse quantile DataFrame into shared ``p`` grid and value array ``q``.

        Returns
        -------
        p : 1D np.ndarray, shape (n_p,)
            the shared quantile points, sorted ascending
        q : 3D np.ndarray, shape (n_rows, n_cols, n_p)
            quantile values, ``q[m, n, k]`` is the value at quantile point ``p[k]``
            for instance row ``m`` and column ``n``
        row_index : pd.Index
            instance (sample) index
        col_index : pd.Index
            column (variable) index
        """
        alphas = sorted(quantiles.index.get_level_values(0).unique())
        samples = quantiles.index.get_level_values(1).unique()
        col_index = quantiles.columns

        p = np.array(alphas, dtype=float)
        q = np.stack(
            [quantiles.loc[a].reindex(samples).to_numpy() for a in alphas], axis=-1
        )

        return p, q, pd.Index(samples), col_index

    def _build_delegate(self, p, q, tails, index, columns):
        """Construct the underlying distribution to delegate to."""
        # interior bins: edges are the quantile values, masses are the p-gaps
        interior_mass = np.diff(p)
        interior_total = p[-1] - p[0]
        lower_mass = float(p[0])
        upper_mass = float(1.0 - p[-1])

        if isinstance(tails, str) and tails in ("uniform", "none"):
            bins, bin_mass = self._build_histogram_params(
                q, tails, interior_mass, lower_mass, upper_mass
            )
            return Histogram(bins=bins, bin_mass=bin_mass, index=index, columns=columns)

        bins, bin_mass = self._build_histogram_params(
            q, "none", interior_mass, lower_mass, upper_mass
        )
        interior = Histogram(bins=bins, bin_mass=bin_mass, index=index, columns=columns)

        components = [("interior", interior)]
        weights = [interior_total]

        q_low = q[:, :, 0]
        q_high = q[:, :, -1]

        lower, upper = self._build_tail_components(
            tails, q_low, q_high, lower_mass, upper_mass, index, columns
        )
        if lower is not None:
            components.append(("lower_tail", lower))
            weights.append(lower_mass)
        if upper is not None:
            components.append(("upper_tail", upper))
            weights.append(upper_mass)

        return Mixture(distributions=components, weights=weights)

    def _build_histogram_params(self, q, tails, interior_mass, lower_mass, upper_mass):
        """Build per-cell``bins``and``bin_mass``for the pure-histogram tail modes."""
        n_rows, n_cols, _ = q.shape

        def cell(qvals):
            edges, masses = list(qvals), list(interior_mass)
            if tails == "uniform" and lower_mass > 0:
                d = interior_mass[0] / (qvals[1] - qvals[0])
                edges, masses = [qvals[0] - lower_mass / d] + edges, [
                    lower_mass
                ] + masses
            if tails == "uniform" and upper_mass > 0:
                d = interior_mass[-1] / (qvals[-1] - qvals[-2])
                edges, masses = edges + [qvals[-1] + upper_mass / d], masses + [
                    upper_mass
                ]
            # renormalize to one (exact for "uniform", drops tails for "none")
            masses = np.array(masses) / np.sum(masses)
            return np.array(edges), masses

        cells = [[cell(q[m, n]) for n in range(n_cols)] for m in range(n_rows)]
        bins = [[edges for edges, _ in row] for row in cells]
        bin_mass = [[masses for _, masses in row] for row in cells]
        return bins, bin_mass

    def _build_tail_components(
        self, tails, q_low, q_high, lower_mass, upper_mass, index, columns
    ):
        """Build lower/upper tail components for ``"mass"`` and distribution tails."""
        lower = None
        upper = None

        if tails == "mass":
            if lower_mass > 0:
                lower = Delta(c=q_low, index=index, columns=columns)
            if upper_mass > 0:
                upper = Delta(c=q_high, index=index, columns=columns)
            return lower, upper

        # distribution tails: cut it at zero and shift each half onto a tail
        # (rebuilt with the QPD's index/columns so it broadcasts to the right shape)
        if isinstance(tails, BaseDistribution):
            base_params = tails.get_params(deep=False)
            base = type(tails)(**{**base_params, "index": index, "columns": columns})
            if lower_mass > 0:
                lower_half = TruncatedDistribution(base, upper=0)
                lower = MeanScale(
                    d=lower_half, mu=q_low, sigma=1, index=index, columns=columns
                )
            if upper_mass > 0:
                upper_half = TruncatedDistribution(base, lower=0)
                upper = MeanScale(
                    d=upper_half, mu=q_high, sigma=1, index=index, columns=columns
                )
            return lower, upper

        raise ValueError(
            "tails must be one of 'uniform', 'none', 'mass', "
            "or an skpro BaseDistribution instance, "
            f"but found {tails!r}"
        )

    def _iloc(self, rowidx=None, colidx=None):
        # keep the full quantiles and narrow via index/columns labels, which the
        # constructor selects in a duplicate- and order-safe way
        subs_rowidx = self.index if rowidx is None else self.index[rowidx]
        subs_colidx = self.columns if colidx is None else self.columns[colidx]

        return HistogramQPD(
            self.quantiles,
            tails=self.tails,
            index=subs_rowidx,
            columns=subs_colidx,
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from skpro.distributions.normal import Normal

        quantile_idx = pd.MultiIndex.from_product(
            [[0.1, 0.5, 0.9], [0, 1, 2]], names=["quantile", "sample"]
        )
        quantiles = pd.DataFrame(
            [[0, 1], [2, 3], [4, 5], [1, 2], [4, 5], [7, 8], [2, 3], [6, 7], [10, 11]],
            index=quantile_idx,
            columns=["a", "b"],
        )
        index = pd.RangeIndex(3)
        columns = pd.Index(["a", "b"])

        params1 = {
            "quantiles": quantiles,
            "tails": "uniform",
            "index": index,
            "columns": columns,
        }
        params2 = {
            "quantiles": quantiles,
            "tails": "none",
            "index": index,
            "columns": columns,
        }
        params3 = {
            "quantiles": quantiles,
            "tails": "mass",
            "index": index,
            "columns": columns,
        }
        params4 = {
            "quantiles": quantiles,
            "tails": Normal(mu=0, sigma=1),
            "index": index,
            "columns": columns,
        }
        return [params1, params2, params3, params4]
