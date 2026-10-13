# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Categorical (finite discrete) probability distribution."""

__author__ = ["Atishyy27"]

from numbers import Number

import numpy as np
import pandas as pd

from skpro.distributions.base import _BaseArrayDistribution


class Categorical(_BaseArrayDistribution):
    r"""Categorical distribution, i.e., a finite discrete distribution.

    The categorical distribution places probability mass ``p[k]`` on the
    support point ``values[k]``, i.e.,

    .. math:: P(X = v_k) = p_k, \quad k = 0, \dots, K-1,

    with :math:`p_k \geq 0` and :math:`\sum_k p_k = 1`.

    Support points default to the integer category codes ``0, ..., K-1``,
    where ``K = len(p)``. With explicit ``values``, this is the finite analogue
    of ``scipy.stats.rv_discrete`` with ``values=(xk, pk)``.

    As in ``rv_discrete``, ``pmf`` is non-zero only at points equal to a support
    point, compared as floats. ``ppf`` returns the smallest support point ``v``
    with ``cdf(v) >= q``, so for ``q`` in ``(0, 1]`` a support point with zero
    mass is never returned.

    Parameters
    ----------
    p : 1D array-like, or 2D list of 1D array-like
        Probability masses, finite, non-negative and summing to 1.

        1. 1D array-like of length ``K`` defines a single (scalar) distribution.

           Example: ``p = [0.2, 0.5, 0.3]``

        2. 2D list of size ``m x n``, where each entry is a 1D array-like as in
           case 1, defines an array-valued distribution with entry-wise masses.
           Every row must have the same number of entries ``n``; the entries
           themselves may have different support sizes.

           Example::

               p = [
                   [[0.2, 0.5, 0.3], [0.1, 0.9]],
                   [[0.6, 0.4], [0.25, 0.25, 0.5]],
               ]

    values : None, 1D array-like, or 2D list of 1D array-like, default=None
        Finite, unique numeric support points, one per entry of ``p``.
        Need not be sorted.

        1. ``None`` uses the integer category codes ``0, ..., K-1`` for every
           entry.

        2. 1D array-like is used as the common support of every entry
           (all entries must then have the same support size).

           Example: ``values = [10, 20, 30]``

        3. 2D list of size ``m x n`` of 1D array-like, matching ``p``
           entry-wise, for the array-valued case with entry-wise supports.

    index : pd.Index, optional, default = RangeIndex
        Only for the array-valued case.
    columns : pd.Index, optional, default = RangeIndex
        Only for the array-valued case.

    Examples
    --------
    >>> from skpro.distributions.categorical import Categorical
    >>>
    >>> d = Categorical(p=[0.25, 0.5, 0.25])
    >>> d.mean()
    1.0

    >>> d = Categorical(p=[0.25, 0.5, 0.25], values=[10, 20, 30])
    >>> d.cdf(20)
    0.75
    """

    _tags = {
        "authors": ["Atishyy27"],
        "capabilities:approx": [],
        "capabilities:exact": ["mean", "var", "pmf", "log_pmf", "cdf", "ppf", "energy"],
        "distr:measuretype": "discrete",
        "distr:paramtype": "parametric",
        "broadcast_init": "on",
    }

    def __init__(self, p, values=None, index=None, columns=None):
        self.p = p
        self.values = values

        self._p_res, self._v_res, self._nested = self._resolve_params(p, values)
        self._check_params()

        # masses sum to 1 within tolerance; rescale them so that every method
        # uses the same distribution
        if self._nested:
            self._p_res = [[self._rescale(pk) for pk in row] for row in self._p_res]
        else:
            self._p_res = self._rescale(self._p_res)

        if self._nested:
            # base broadcasting needs concrete index and columns
            if index is None:
                index = pd.RangeIndex(len(self._p_res))
            if columns is None:
                columns = pd.RangeIndex(len(self._p_res[0]))
        elif index is not None or columns is not None:
            raise ValueError(
                "index and columns are only supported for the array-valued case, "
                "where p is a 2D list of 1D probability vectors; got a 1D p"
            )

        super().__init__(index=index, columns=columns)

    # parameter handling
    # ------------------
    @staticmethod
    def _is_number(x):
        if isinstance(x, np.ndarray):
            return x.ndim == 0
        return isinstance(x, (Number, np.number, np.bool_))

    @staticmethod
    def _rescale(pk):
        total = pk.sum()
        # masses that already sum to 1 up to rounding are kept as passed
        if abs(total - 1.0) <= 1e-12:
            return pk
        return pk / total

    @staticmethod
    def _one_entry(pk, vk):
        """Coerce one (masses, support) pair to float arrays sorted by support."""
        pk = np.asarray(pk, dtype=float)
        if pk.ndim != 1:
            raise ValueError(f"each entry of p must be 1D, got {pk.ndim}D masses")
        if vk is None:
            vk = np.arange(len(pk))
        vk = np.asarray(vk, dtype=float)
        if vk.ndim != 1:
            raise ValueError(f"each entry of values must be 1D, got {vk.ndim}D values")
        if len(vk) != len(pk):
            raise ValueError(
                f"values must have the same length as p, got {len(vk)} and {len(pk)}"
            )
        order = np.argsort(vk)
        return pk[order], vk[order]

    @classmethod
    def _resolve_params(cls, p, values):
        """Return masses and support with defaults applied, sorted by support.

        Returns
        -------
        p : 1D np.ndarray, or 2D list of 1D np.ndarray
        values : 1D np.ndarray, or 2D list of 1D np.ndarray, same nesting as ``p``
        nested : bool, True iff array-valued
        """
        if isinstance(p, pd.Series):
            p = p.to_numpy()
        if isinstance(values, pd.Series):
            values = values.to_numpy()

        if cls._is_number(p) or isinstance(p, str):
            raise ValueError("p must be 1D, or a 2D list of 1D probability vectors")
        if len(p) == 0:
            raise ValueError("p must have at least one entry")

        if cls._is_number(p[0]):
            pk, vk = cls._one_entry(p, values)
            return pk, vk, False

        nesting_msg = (
            "array-valued p must be nested as a 2D list of 1D probability "
            "vectors, got a scalar at p{}; for a single distribution, pass a 1D p"
        )
        n_cols = len(p[0])
        if n_cols == 0:
            raise ValueError("p must have at least one entry")
        for i, row in enumerate(p):
            if cls._is_number(row):
                raise ValueError(nesting_msg.format(f"[{i}]"))
            if len(row) != n_cols:
                raise ValueError(
                    "every row of p must have the same number of entries, "
                    f"got {n_cols} in row 0 and {len(row)} in row {i}"
                )
            for j, entry in enumerate(row):
                if cls._is_number(entry):
                    raise ValueError(nesting_msg.format(f"[{i}][{j}]"))

        shared_values = values is not None and cls._is_number(values[0])
        if values is not None and not shared_values:
            if len(values) != len(p) or any(
                cls._is_number(values[i]) or len(values[i]) != len(p[i])
                for i in range(len(p))
            ):
                raise ValueError("entry-wise values must have the same nesting as p")

        p_out, v_out = [], []
        for i in range(len(p)):
            p_row, v_row = [], []
            for j in range(n_cols):
                if values is None:
                    vk = None
                elif shared_values:
                    vk = values
                else:
                    vk = values[i][j]
                pk, vk = cls._one_entry(p[i][j], vk)
                p_row.append(pk)
                v_row.append(vk)
            p_out.append(p_row)
            v_out.append(v_row)
        return p_out, v_out, True

    def _resolve(self):
        """Return the masses and support resolved in ``__init__``."""
        return self._p_res, self._v_res, self._nested

    def _check_params(self):
        """Validate masses and support, raising ValueError on invalid input."""
        p, values, nested = self._resolve()

        def _check_one(pk, vk, where):
            if len(pk) == 0:
                raise ValueError(f"p{where} must have at least one entry")
            if not np.all(np.isfinite(pk)):
                raise ValueError(f"p{where} must be finite")
            if not np.all(np.isfinite(vk)):
                raise ValueError(f"values{where} must be finite")
            if np.any(pk < 0):
                raise ValueError(f"p{where} must be non-negative")
            if not np.isclose(pk.sum(), 1.0):
                raise ValueError(f"p{where} must sum to 1, got sum {pk.sum()}")
            if len(np.unique(vk)) != len(vk):
                raise ValueError(f"values{where} must be unique")

        if not nested:
            _check_one(p, values, "")
            return None

        for i in range(len(p)):
            for j in range(len(p[i])):
                _check_one(p[i][j], values[i][j], f"[{i}][{j}]")

    def _get_dist_params(self):
        """Return resolved parameters, for broadcasting and subsetting.

        A shared 1D ``values`` is expanded to match ``p`` entry-wise, so that
        subsetting keeps each entry attached to its own support.
        """
        p, values, _ = self._resolve()
        return {"p": p, "values": values}

    # per-entry formulae
    # ------------------
    @staticmethod
    def _cdf_steps(pk):
        """Return the cdf at each support point, with the last step exactly 1."""
        cs = np.cumsum(pk)
        return cs / cs[-1]

    @staticmethod
    def _mean_one(pk, vk):
        return float(np.dot(pk, vk))

    @staticmethod
    def _var_one(pk, vk):
        # centre on the smallest support point before squaring, so that large
        # support values do not cancel out the variance
        vc = vk - vk[0]
        mean_c = np.dot(pk, vc)
        return float(np.dot(pk, (vc - mean_c) ** 2))

    @staticmethod
    def _pmf_one(pk, vk, x):
        if np.isnan(x):
            return np.nan
        return float(pk[vk == x].sum())

    @classmethod
    def _cdf_one(cls, pk, vk, x):
        if np.isnan(x):
            return np.nan
        k = int(np.searchsorted(vk, x, side="right")) - 1
        if k < 0:
            return 0.0
        return float(cls._cdf_steps(pk)[k])

    @classmethod
    def _ppf_one(cls, pk, vk, q):
        if not 0.0 <= q <= 1.0:  # also False for nan
            return np.nan
        k = int(np.searchsorted(cls._cdf_steps(pk), q, side="left"))
        return float(vk[min(k, len(vk) - 1)])

    @classmethod
    def _energy_self_one(cls, pk, vk):
        # E|X-Y| = 2 * integral of F(1-F), which for a step cdf is a sum over
        # the gaps between consecutive support points; linear in K
        cdf = cls._cdf_steps(pk)[:-1]
        return float(2 * np.sum(cdf * (1 - cdf) * np.diff(vk)))

    @staticmethod
    def _energy_x_one(pk, vk, x):
        return float(np.dot(pk, np.abs(vk - x)))

    def _apply(self, fun, x=None):
        """Apply a per-entry formula over the scalar or array-valued case."""
        p, values, nested = self._resolve()
        if not nested:
            if x is None:
                return fun(p, values)
            return fun(p, values, x)

        x = None if x is None else np.asarray(x)
        out = []
        for i in range(len(p)):
            row = []
            for j in range(len(p[i])):
                if x is None:
                    row.append(fun(p[i][j], values[i][j]))
                else:
                    row.append(fun(p[i][j], values[i][j], x[i][j]))
            out.append(row)
        return np.array(out)

    # BaseDistribution interface
    # --------------------------
    def _mean(self):
        """Return expected value of the distribution.

        Returns
        -------
        float, or 2D np.ndarray, same shape as ``self``
            expected value of distribution (entry-wise)
        """
        return self._apply(self._mean_one)

    def _var(self):
        r"""Return element/entry-wise variance of the distribution.

        Returns
        -------
        float, or 2D np.ndarray, same shape as ``self``
            variance of the distribution (entry-wise)
        """
        return self._apply(self._var_one)

    def _pmf(self, x):
        """Probability mass function.

        Parameters
        ----------
        x : float, or 2D np.ndarray, same shape as ``self``
            values to evaluate the pmf at

        Returns
        -------
        float, or 2D np.ndarray, same shape as ``self``
            pmf values at the given points, 0 unless ``x`` equals a support point
        """
        return self._apply(self._pmf_one, x)

    def _log_pmf(self, x):
        """Logarithmic probability mass function.

        Parameters
        ----------
        x : float, or 2D np.ndarray, same shape as ``self``
            values to evaluate the log pmf at

        Returns
        -------
        float, or 2D np.ndarray, same shape as ``self``
            log pmf values at the given points, ``-inf`` outside the support
        """
        with np.errstate(divide="ignore"):
            return np.log(self._pmf(x))

    def _cdf(self, x):
        """Cumulative distribution function.

        Parameters
        ----------
        x : float, or 2D np.ndarray, same shape as ``self``
            values to evaluate the cdf at

        Returns
        -------
        float, or 2D np.ndarray, same shape as ``self``
            cdf values at the given points
        """
        return self._apply(self._cdf_one, x)

    def _ppf(self, p):
        """Quantile function = percent point function = inverse cdf.

        Returns the smallest support point ``v`` with ``cdf(v) >= p``.

        Parameters
        ----------
        p : float, or 2D np.ndarray, same shape as ``self``
            values to evaluate the ppf at

        Returns
        -------
        float, or 2D np.ndarray, same shape as ``self``
            ppf values at the given points, ``nan`` outside ``[0, 1]``
        """
        return self._apply(self._ppf_one, p)

    def _energy_self(self):
        r"""Energy of self, w.r.t. self.

        :math:`\mathbb{E}[|X-Y|] = \sum_{i,j} p_i p_j |v_i - v_j|`,
        where :math:`X, Y` are i.i.d. copies of self. Computed in linear time as
        :math:`2 \sum_{k} F_k (1 - F_k) (v_{k+1} - v_k)` over sorted support.

        Returns
        -------
        float, or 1D np.ndarray of length ``self.shape[0]``
            energy values, summed over columns in the array-valued case
        """
        res = self._apply(self._energy_self_one)
        if np.ndim(res) > 0:
            res = np.sum(res, axis=1)
        return res

    def _energy_x(self, x):
        r"""Energy of self, w.r.t. a constant frame x.

        :math:`\mathbb{E}[|X-x|] = \sum_i p_i |v_i - x|`,
        where :math:`X` is a copy of self, and :math:`x` is a constant.

        Parameters
        ----------
        x : float, or 2D np.ndarray, same shape as ``self``
            values to compute energy w.r.t. to

        Returns
        -------
        float, or 1D np.ndarray of length ``self.shape[0]``
            energy values, summed over columns in the array-valued case
        """
        res = self._apply(self._energy_x_one, x)
        if np.ndim(res) > 0:
            res = np.sum(res, axis=1)
        return res

    def _sample(self, n_samples=None):
        """Sample from the distribution, by inverse transform on the support.

        Draws all samples of each entry at once.

        Parameters
        ----------
        n_samples : int, optional, default = None
            number of samples to draw from the distribution

        Returns
        -------
        pd.DataFrame, or float if ``self`` is scalar and ``n_samples`` is None
            samples from the distribution, in the format of ``sample``
        """
        p, values, nested = self._resolve()
        n = 1 if n_samples is None else n_samples

        def draw(pk, vk):
            # u in (0, 1], so that a leading zero-mass point is never drawn
            u = 1 - np.random.uniform(size=n)
            k = np.searchsorted(self._cdf_steps(pk), u, side="left")
            return vk[np.minimum(k, len(vk) - 1)]

        if not nested:
            spl = draw(p, values)
            if n_samples is None:
                return float(spl[0])
            return pd.DataFrame(spl.tolist())

        spl = np.empty((n, len(p), len(p[0])))
        for i in range(len(p)):
            for j in range(len(p[0])):
                spl[:, i, j] = draw(p[i][j], values[i][j])

        frames = [
            pd.DataFrame(spl[s], index=self.index, columns=self.columns)
            for s in range(n)
        ]
        if n_samples is None:
            return frames[0]
        return pd.concat(frames, keys=range(n))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # scalar case, default integer support
        params1 = {"p": [0.2, 0.5, 0.3]}
        # scalar case, explicit unsorted support
        params2 = {"p": [0.3, 0.1, 0.6], "values": [30, 10, 20]}
        # array-valued case, shared support
        params3 = {
            "p": [
                [[0.2, 0.5, 0.3], [0.1, 0.1, 0.8]],
                [[1 / 3, 1 / 3, 1 / 3], [0.05, 0.9, 0.05]],
                [[0.6, 0.3, 0.1], [0.25, 0.25, 0.5]],
            ],
            "values": [0, 1, 2],
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }
        # array-valued case, entry-wise supports of different sizes
        params4 = {
            "p": [
                [[0.5, 0.5], [0.2, 0.3, 0.5]],
                [[0.1, 0.9], [0.25, 0.25, 0.25, 0.25]],
                [[0.7, 0.3], [1.0]],
            ],
            "values": [
                [[0, 1], [1, 2, 3]],
                [[-1, 1], [0, 10, 20, 30]],
                [[2.5, 5.0], [42]],
            ],
            "index": pd.Index([1, 2, 5]),
            "columns": pd.Index(["a", "b"]),
        }

        return [params1, params2, params3, params4]
