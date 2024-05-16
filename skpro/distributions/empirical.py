# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Empirical distribution."""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from skpro.distributions.base import BaseDistribution


class Empirical(BaseDistribution):
    r"""Empirical distribution, or weighted sum of delta distributions.

    This distribution represents an empirical distribution, or, more generally,
    a weighted sum of delta distributions.

    The distribution is parameterized by support in ``spl``, and optionally
    weights in ``weights``.

    For the scalar case, the distribution is parameterized as follows:
    let :math:`s_i, i = 1 \dots N` the entries of ``spl``,
    and :math:`w_i, i = 1 \dots N` the entries of ``weights``; if ``weights=None``,
    by default we define :math:`p_i = \frac{1}{N}`, otherwise we
    define :math:`p_i := \frac{w_i}{\sum_{i=1}^N w_i}`

    The distribution is the unique distribution that takes value :math:`s_i` with
    probability :math:`p_i`. In particluar, if ``weights`` was ``None``,
    the distribution is the uniform distribution supported on the :math:`s_i`.

    Parameters
    ----------
    spl : pd.DataFrame
        empirical sample; for scalar distributions, rows are samples;
        for dataframe-like distributions,
        first (lowest) index is sample, further indices are instance indices
    weights : pd.Series, with same index and length as spl, optional, default=None
        if not passed, ``spl`` is assumed to be unweighted
    time_indep : bool, optional, default=True
        if True, ``sample`` will sample individual instance indices independently
        if False, ``sample`` will sample entire instances from ``spl``
    index : pd.Index, optional, default = RangeIndex
    columns : pd.Index, optional, default = RangeIndex

    Example
    -------
    >>> import pandas as pd
    >>> from skpro.distributions.empirical import Empirical

    >>> spl_idx = pd.MultiIndex.from_product(
    ...     [[0, 1], [0, 1, 2]], names=["sample", "time"]
    ... )
    >>> spl = pd.DataFrame(
    ...     [[0, 1], [2, 3], [10, 11], [6, 7], [8, 9], [4, 5]],
    ...     index=spl_idx,
    ...     columns=["a", "b"],
    ... )
    >>> dist = Empirical(spl)
    >>> empirical_sample = dist.sample(3)

    scalar distribution:
    >>> spl = pd.Series([1, 2, 3, 4, 3])
    >>> dist = Empirical(spl)
    >>> empirical_sample = dist.sample(3)
    """

    _tags = {
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "energy", "cdf", "ppf"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "nonparametric",
    }

    def __init__(self, spl, weights=None, time_indep=True, index=None, columns=None):
        self.spl = spl
        self.weights = weights
        self.time_indep = time_indep

        index, columns = self._init_index(index, columns)

        super().__init__(index=index, columns=columns)

        # initialized sorted samples
        self._init_sorted()

    def _init_index(self, index, columns):
        """Initialize index and columns.

        Sets the following attributes:

        * ``_spl_indices`` - unique index for samples
        * ``_shape`` - shape of self - 0D or 2D
        * ``_N`` - number of samples
        * only if array distribution: ``_instances``,
          coerced index of ``self``, from ``spl`` index
        """
        spl = self.spl

        is_scalar = not isinstance(spl.index, pd.MultiIndex)
        is_scalar = is_scalar and (spl.ndim <= 1 or spl.shape[1] == 1)

        if is_scalar:
            self._shape = ()
            _spl_indices = spl.index
            self._spl_indices = _spl_indices
            self._N = len(_spl_indices)
            return None, None

        _instances = spl.index.droplevel(0).unique()
        _spl_indices = spl.index.get_level_values(0).unique()
        self._instances = _instances
        self._spl_indices = _spl_indices
        self._N = len(_spl_indices)

        if index is None:
            index = _instances
        if columns is None:
            columns = spl.columns

        self._shape = (len(index), len(columns))

        return index, columns

    def _init_sorted(self):
        """Initialize sorted version of spl."""
        if self.ndim == 0:
            spl = self.spl.values.flatten()
            sorter = np.argsort(spl)
            spl_sorted = spl[sorter]
            if self.weights is not None:
                weights_sorted = self.weights.values.flatten()[sorter]
            else:
                weights_sorted = np.ones_like(spl)
            self._sorted = spl_sorted
            self._weights = weights_sorted
            return None

        times = self._instances
        cols = self.columns

        sorted = {}
        weights = {}
        for t in times:
            sorted[t] = {}
            weights[t] = {}
            for col in cols:
                sl = (slice(None),) + self._coerce_tuple(t)
                spl_t = self.spl.loc[sl, col].values
                sorter = np.argsort(spl_t)
                spl_t_sorted = spl_t[sorter]
                sorted[t][col] = spl_t_sorted
                if self.weights is not None:
                    weights_t = self.weights.loc[sl].values
                    weights_t_sorted = weights_t[sorter]
                    weights[t][col] = weights_t_sorted
                else:
                    ones = np.ones(len(spl_t_sorted))
                    weights[t][col] = ones

        self._sorted = sorted
        self._weights = weights

    def _coerce_tuple(self, x):
        if not isinstance(x, tuple):
            x = (x,)
        return x

    def _apply_per_ix(self, func, params, x=None):
        """Apply function per index."""
        sorted = self._sorted
        weights = self._weights

        if self.ndim == 0:
            return func(spl=sorted, weights=weights, x=x, **params)

        index = self.index
        cols = self.columns

        res = pd.DataFrame(index=index, columns=cols)
        for i, ix in enumerate(index):
            for j, col in enumerate(cols):
                spl_t = sorted[ix][col]
                weights_t = weights[ix][col]
                if x is None:
                    x_t = None
                elif hasattr(x, "loc"):
                    x_t = x.loc[ix, col]
                else:
                    x_t = x[i, j]
                res.at[ix, col] = func(spl=spl_t, weights=weights_t, x=x_t, **params)
        return res.apply(pd.to_numeric)

    def _slice_ix(self, obj, ix):
        """Slice obj by index ix, applied to MultiIndex levels 1 ... last.

        obj is assumed to have MultiIndex, and slicing occurs on the
        last levels, 1 ... last.

        ix can be a simple index or MultiIndex,
        same number of levels as obj.index.droplevel(0).

        This utility function is needed since pandas cannot do this?

        Parameters
        ----------
        obj : pd.DataFrame or pd.Series, with pd.MultiIndex
            object to slice
        ix : pd.Index or pd.MultiIndex
            index to slice by, loc references

        Returns
        -------
        pd.DataFrame or pd.Series, same type as obj
            obj sliced by levels 1 ... last, subset to levels in ix
        """
        if not isinstance(ix, pd.MultiIndex):
            if isinstance(obj, pd.DataFrame):
                return obj.loc[(slice(None), ix), :]
            else:
                return obj.loc[(slice(None), ix)]

        obj_ix = obj.index
        obj_vals = obj_ix.get_level_values(0).unique()
        if isinstance(ix, pd.MultiIndex):
            prod_ix = [(v,) + i for v in obj_vals for i in ix]
        else:
            prod_ix = [(v,) + (i,) for v in obj_vals for i in ix]
        prod_ix = pd.MultiIndex.from_tuples(prod_ix)

        return obj.loc[prod_ix]

    def _iloc(self, rowidx=None, colidx=None):
        if is_scalar_notnone(rowidx) and is_scalar_notnone(colidx):
            return self._iat(rowidx, colidx)
        if is_scalar_notnone(rowidx):
            rowidx = pd.Index([rowidx])
        if is_scalar_notnone(colidx):
            colidx = pd.Index([colidx])

        index = self.index
        columns = self.columns
        weights = self.weights

        spl_subset = self.spl

        if rowidx is not None:
            rowidx_loc = index[rowidx]
            # subset multiindex to rowidx by last level
            spl_subset = self._slice_ix(self.spl, rowidx_loc)
            if weights is not None:
                weights_subset = self._slice_ix(weights, rowidx_loc)
            else:
                weights_subset = None
            subs_rowidx = index[rowidx]
        else:
            subs_rowidx = index
            weights_subset = weights

        if colidx is not None:
            spl_subset = spl_subset.iloc[:, colidx]
            subs_colidx = columns[colidx]
        else:
            subs_colidx = columns

        return Empirical(
            spl_subset,
            weights=weights_subset,
            time_indep=self.time_indep,
            index=subs_rowidx,
            columns=subs_colidx,
        )

    def _iat(self, rowidx=None, colidx=None):
        if rowidx is None or colidx is None:
            raise ValueError("iat method requires both row and column index")
        self_subset = self.iloc[[rowidx], [colidx]]
        spl_subset = self_subset.spl.droplevel(0)
        if self.weights is not None:
            wts_subset = self_subset.weights.droplevel(0)
        else:
            wts_subset = None

        subset_params = {"spl": spl_subset, "weights": wts_subset}
        return type(self)(**subset_params)

    def _energy_default(self, x=None):
        r"""Energy of self, w.r.t. self or a constant frame x.

        Let :math:`X, Y` be i.i.d. random variables with the distribution of `self`.

        If `x` is `None`, returns :math:`\mathbb{E}[|X-Y|]` (per row), "self-energy".
        If `x` is passed, returns :math:`\mathbb{E}[|X-x|]` (per row), "energy wrt x".

        Parameters
        ----------
        x : None or pd.DataFrame, optional, default=None
            if pd.DataFrame, must have same rows and columns as `self`

        Returns
        -------
        pd.DataFrame with same rows as `self`, single column `"energy"`
        each row contains one float, self-energy/energy as described above.
        """
        energy_arr = self._apply_per_ix(_energy_np, {"assume_sorted": True}, x=x)
        if energy_arr.ndim > 0:
            energy_arr = np.sum(energy_arr, axis=1)
        return energy_arr

    def _mean(self):
        r"""Return expected value of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns the expectation :math:`\mathbb{E}[X]`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        expected value of distribution (entry-wise)
        """
        spl = self.spl

        # scalar case
        if self.ndim == 0:
            spl = spl.values.flatten()
            if self.weights is None:
                return np.mean(spl)
            else:
                return np.average(spl, weights=self.weights)

        # dataframe case
        groupby_levels = list(range(1, spl.index.nlevels))
        if self.weights is None:
            mean_df = spl.groupby(level=groupby_levels, sort=False).mean()
        else:
            mean_df = spl.groupby(level=groupby_levels, sort=False).apply(
                lambda x: np.average(x, weights=self.weights.loc[x.index], axis=0)
            )
            mean_df = pd.DataFrame(mean_df.tolist(), index=mean_df.index)
            mean_df.columns = spl.columns

        mean_df = mean_df.loc[self.index]  # ensure consistent sorting
        return mean_df

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Let :math:`X` be a random variable with the distribution of `self`.
        Returns :math:`\mathbb{V}[X] = \mathbb{E}\left(X - \mathbb{E}[X]\right)^2`

        Returns
        -------
        pd.DataFrame with same rows, columns as `self`
        variance of distribution (entry-wise)
        """
        spl = self.spl
        N = self._N

        # scalar case
        if self.ndim == 0:
            spl = spl.values.flatten()
            if self.weights is None:
                return np.var(spl, ddof=0)
            else:
                mean = self.mean()
                var = np.average((spl - mean) ** 2, weights=self.weights)
                return var

        # dataframe case
        groupby_levels = list(range(1, spl.index.nlevels))
        if self.weights is None:
            var_df = spl.groupby(level=groupby_levels, sort=False).var(ddof=0)
        else:
            mean = self.mean()
            means = pd.concat([mean] * N, axis=0, keys=self._spl_indices)
            var_df = spl.groupby(level=groupby_levels, sort=False).apply(
                lambda x: np.average(
                    (x - means.loc[x.index]) ** 2,
                    weights=self.weights.loc[x.index],
                    axis=0,
                )
            )
            var_df = pd.DataFrame(
                var_df.tolist(), index=var_df.index, columns=spl.columns
            )

        var_df = var_df.loc[self.index]  # ensure consistent sorting
        return var_df

    def _cdf(self, x):
        """Cumulative distribution function."""
        cdf_val = self._apply_per_ix(_cdf_np, {"assume_sorted": True}, x=x)
        return cdf_val

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf."""
        ppf_val = self._apply_per_ix(_ppf_np, {"assume_sorted": True}, x=p)
        return ppf_val

    def sample(self, n_samples=None):
        """Sample from the distribution.

        Parameters
        ----------
        n_samples : int, optional, default = None

        Returns
        -------
        if `n_samples` is `None`:
        returns a sample that contains a single sample from `self`,
        in `pd.DataFrame` mtype format convention, with `index` and `columns` as `self`
        if n_samples is `int`:
        returns a `pd.DataFrame` that contains `n_samples` i.i.d. samples from `self`,
        in `pd-multiindex` mtype format convention, with same `columns` as `self`,
        and `MultiIndex` that is product of `RangeIndex(n_samples)` and `self.index`
        """
        # for now, always defaulting to the standard logic
        # todo: address issue #283
        if self.ndim >= 0:
            return super().sample(n_samples=n_samples)

        spl = self.spl
        timestamps = self._instances
        weights = self.weights

        if n_samples is None:
            n_samples = 1
            n_samples_was_none = True
        else:
            n_samples_was_none = False
        smpls = []

        for _ in range(n_samples):
            smpls_i = []
            for t in timestamps:
                spl_from = spl.loc[(slice(None), t), :]
                if weights is not None:
                    spl_weights = weights.loc[(slice(None), t)].values
                else:
                    spl_weights = None
                spl_time = spl_from.sample(n=1, replace=True, weights=spl_weights)
                spl_time = spl_time.droplevel(0)
                smpls_i.append(spl_time)
            spl_i = pd.concat(smpls_i, axis=0)
            smpls.append(spl_i)

        spl = pd.concat(smpls, axis=0, keys=range(n_samples))
        if n_samples_was_none:
            spl = spl.droplevel(0)

        return spl

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # params1 is a DataFrame with simple row multiindex
        spl_idx = pd.MultiIndex.from_product(
            [[0, 1], [0, 1, 2]], names=["sample", "time"]
        )
        spl = pd.DataFrame(
            [[0, 1], [2, 3], [10, 11], [6, 7], [8, 9], [4, 5]],
            index=spl_idx,
            columns=["a", "b"],
        )
        params1 = {
            "spl": spl,
            "weights": None,
            "time_indep": True,
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a", "b"]),
        }

        # params2 is weighted
        params2 = {
            "spl": spl,
            "weights": pd.Series([0.5, 0.5, 0.5, 1, 1, 1.1], index=spl_idx),
            "time_indep": False,
            "index": pd.RangeIndex(3),
            "columns": pd.Index(["a", "b"]),
        }

        # params3 is scalar, unweighted
        spl_scalar = pd.Series([1, 2, 3, 4, 3])
        params3 = {"spl": spl_scalar}

        # params4 is scalar, weighted
        spl_scalar = pd.DataFrame([1, 2, 3, 4, 3])
        wts_scalar = pd.Series([0.2, 0.2, 0.3, 0.3, 0.1])
        params4 = {"spl": spl_scalar, "weights": wts_scalar}

        # hierarchical MultiIndex, important for sktime
        spl_idx = pd.MultiIndex.from_product(
            [[0, 1], [0, 1, 2], [0, 1]], names=["sample", "instance", "time"]
        )
        param5 = {"spl": pd.DataFrame(np.random.rand(12, 2), index=spl_idx)}

        # hierarchical MultiIndex, important for sktime, weighted
        spl_idx = pd.MultiIndex.from_product(
            [[0, 1], [0, 1, 2], [0, 1]], names=["sample", "instance", "time"]
        )
        weights = pd.Series(list(range(1, 13)), index=spl_idx)
        spl = pd.DataFrame(np.random.rand(12, 2), index=spl_idx)
        param6 = {"spl": spl, "weights": weights}

        return [params1, params2, params3, params4, param5, param6]


def _energy_np(spl, x=None, weights=None, assume_sorted=False):
    r"""Compute sample energy, fast numpy based subroutine.

    Let :math:`X` be the random variable with support being
    values of `spl`, with probability weights `weights`.

    This function then returns :math:`\mathbb{E}[|X-Y|]`, with :math:`Y` an
    independent copy of :math:`X`, if `x` is `None`.

    If `x` is passed, returns :math:`\mathbb{E}[|X-x|]`.

    Parameters
    ----------
    spl : 1D np.ndarray
        empirical sample
    x : None or float, optional, default=None
        if None, computes self-energy, if float, computes energy wrt x
    weights : None or 1D np.ndarray, optional, default=None
        if None, computes unweighted energy, if 1D np.ndarray, computes weighted energy
        if not None, must be of same length as ``spl``, needs not be normalized
    assume_sorted : bool, optional, default=False
        if True, assumes that ``spl`` is sorted in ascending order

    Returns
    -------
    float
        energy as described above
    """
    if weights is None:
        weights = np.ones(len(spl))

    if not assume_sorted:
        sorter = np.argsort(spl)
        spl = spl[sorter]
        weights = weights[sorter]

    w_sum = np.sum(weights)
    weights = weights / w_sum

    spl_diff = np.diff(spl)

    if x is None:
        cum_fwd = np.cumsum(weights[:-1])
        cum_back = np.cumsum(weights[1:][::-1])[::-1]
        energy = 2 * np.sum(cum_fwd * cum_back * spl_diff)
    else:
        spl_diff = np.abs(spl - x)
        energy = np.sum(weights * spl_diff)

    return energy


def _cdf_np(spl, x, weights=None, assume_sorted=False):
    """Compute empirical cdf, fast numpy based subroutine.

    Parameters
    ----------
    spl : 1D np.ndarray
        empirical sample
    x : float
        value at which to evaluate cdf
    weights : None or 1D np.ndarray, optional, default=None
        if None, computes unweighted cdf, if 1D np.ndarray, computes weighted cdf
        if not None, must be of same length as ``spl``, needs not be normalized
    assume_sorted : bool, optional, default=False
        if True, assumes that ``spl`` is sorted in ascending order

    Returns
    -------
    cdf_val float
        cdf-value at x
    """
    if weights is None:
        weights = np.ones(len(spl))

    if not assume_sorted:
        sorter = np.argsort(spl)
        spl = spl[sorter]
        weights = weights[sorter]

    w_sum = np.sum(weights)
    weights = weights / w_sum

    weights_select = weights[spl <= x]
    cdf_val = np.sum(weights_select)

    return cdf_val


def _ppf_np(spl, x, weights=None, assume_sorted=False):
    """Compute empirical ppf, fast numpy based subroutine.

    Parameters
    ----------
    spl : 1D np.ndarray
        empirical sample
    x : float
        probability at which to evaluate ppf
    weights : None or 1D np.ndarray, optional, default=None
        if None, computes unweighted ppf, if 1D np.ndarray, computes weighted ppf
        if not None, must be of same length as ``spl``, needs not be normalized
    assume_sorted : bool, optional, default=False
        if True, assumes that ``spl`` is sorted in ascending order

    Returns
    -------
    ppf_val float
        ppf-value at p
    """
    if weights is None:
        weights = np.ones(len(spl))

    if not assume_sorted:
        sorter = np.argsort(spl)
        spl = spl[sorter]
        weights = weights[sorter]

    w_sum = np.sum(weights)
    weights = weights / w_sum

    cum_weights = np.cumsum(weights)
    ix_val = np.searchsorted(cum_weights, x)
    ppf_val = spl[ix_val]

    return ppf_val


def is_scalar_notnone(obj):
    """Check if obj is scalar and not None."""
    return obj is not None and np.isscalar(obj)
