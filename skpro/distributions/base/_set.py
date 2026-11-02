"""Symbolic set representations for distribution supports."""

__author__ = ["khushmagrawal"]

__all__ = [
    "BaseSet",
    "IntervalSet",
    "FiniteSet",
    "IntegerSet",
    "UnionSet",
    "IntersectionSet",
    "EmptySet",
    "RealSet",
]

import numpy as np
import pandas as pd

from skpro.base import BaseObject
from skpro.distributions.base._base import _coerce_to_pd_index_or_none


class BaseSet(BaseObject):
    """Base class for symbolic set representations.

    All set subclasses inherit from ``BaseSet``, gaining access to tags
    (``is_discrete``, ``is_continuous``) and tabular shape information
    (``index``, ``columns``).

    Sets are "shape-aware": a single set object can describe the support
    for an entire tabular distribution by storing its parameters as arrays
    matching the distribution's ``(n_rows, n_cols)`` shape.

    Parameters
    ----------
    index : pd.Index, optional, default = None
        Row index of the distribution this set describes.
    columns : pd.Index, optional, default = None
        Column index of the distribution this set describes.
    """

    _tags = {
        "object_type": "set",
        "is_discrete": False,
        "is_continuous": False,
    }

    def __init__(self, index=None, columns=None):
        self.index = _coerce_to_pd_index_or_none(index)
        self.columns = _coerce_to_pd_index_or_none(columns)

        self._shape: tuple = ()
        if self.index is None and self.columns is None:
            pass
        else:
            row_len = len(self.index) if self.index is not None else 0
            col_len = len(self.columns) if self.columns is not None else 0
            if self.index is not None and self.columns is None:
                col_len = 1
            self._shape = (row_len, col_len)

        super().__init__()

    @property
    def shape(self):
        """Shape of self, a pair (2-tuple) or empty tuple if scalar."""
        return self._shape

    @property
    def ndim(self):
        """Number of dimensions of self. 2 if array, 0 if scalar."""
        return len(self._shape)

    def __contains__(self, item):
        """Check membership using ``in`` operator.

        Delegates to ``self.contains(item)``.
        """
        return self.contains(item)

    def contains(self, x):
        """Check whether ``x`` is contained in the set.

        Parameters
        ----------
        x : scalar, np.ndarray, or pd.DataFrame
            Point(s) to check for membership.

        Returns
        -------
        bool, np.ndarray of bool, or pd.DataFrame of bool
            Membership mask, same shape as ``x`` (or scalar if ``x`` is scalar).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement 'contains'."
        )

    def boundary(self):
        """Return the boundary of the set.

        Returns
        -------
        depends on subclass
            Representation of the set's boundary.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement 'boundary'."
        )

    def __str__(self):
        """Return a human-readable string representation."""
        return self.__class__.__name__

    def __repr__(self):
        """Return a developer-readable string representation."""
        return f"{self.__class__.__name__}()"

    def _align_output(self, res, x):
        """Restore Pandas metadata for x and ensure minimal broadcast shape."""
        if getattr(res, "ndim", 0) == 0 and self.ndim > 0:
            res = np.full(self.shape, res, dtype=bool)

        if isinstance(x, pd.DataFrame):
            if getattr(res, "ndim", 1) == 1 and len(x.columns) > 1:
                res = np.tile(np.asarray(res).reshape(-1, 1), (1, len(x.columns)))
            return pd.DataFrame(res, index=x.index, columns=x.columns)
        elif isinstance(x, pd.Series):
            return pd.Series(res, index=x.index, name=x.name)
        return res

    def _get_bc_params(self, *args, oned_as="col"):
        """Broadcast set parameters against index and columns.

        Mirrors the broadcasting logic of ``BaseDistribution._get_bc_params``:
        reshapes index as column vector ``(-1, 1)`` and columns as row vector,
        then calls ``np.broadcast_arrays`` to expand all parameters to
        the full ``(n_rows, n_cols)`` shape.

        Parameters
        ----------
        args : floats, ints, or array-like
            Set parameters to broadcast.
        oned_as : str, optional, default="col"
            If ``"col"``, 1D arrays are treated as column vectors.

        Returns
        -------
        tuple of np.ndarray
            Broadcasted parameter arrays, same length as ``args``.
        """

        def _to_col(arr):
            if arr.ndim == 1 and oned_as == "col":
                return arr.reshape(-1, 1)
            return arr

        args_np = [_to_col(np.asarray(a)) for a in args]

        if self.index is not None:
            args_np.append(self.index.to_numpy().reshape(-1, 1))
        if self.columns is not None:
            args_np.append(self.columns.to_numpy())

        bc = np.broadcast_arrays(*args_np)
        # return only the parameter arrays, not the index/columns anchors
        return bc[: len(args)]


class IntervalSet(BaseSet):
    r"""Interval of the real line.

    Represents a set of the form :math:`(a, b)`, :math:`[a, b]`,
    :math:`[a, b)`, or :math:`(a, b]`, where
    :math:`a, b \in \mathbb{R} \cup \{-\infty, +\infty\}`.

    Boundaries ``lower`` and ``upper`` can be scalars (applying to all
    entries of a tabular distribution) or arrays matching the
    distribution's ``(n_rows, n_cols)`` shape.

    Parameters
    ----------
    lower : float or array-like
        Lower bound of the interval.
    upper : float or array-like
        Upper bound of the interval.
    interval_type : str, optional, default = "[]"
        Type of interval boundary.
        One of ``"[]"`` (closed), ``"()"`` (open),
        ``"[)"`` (left-closed), ``"(]"`` (right-closed).
        Matches the notation used by ``TruncatedDistribution``.
    index : pd.Index, optional, default = None
    columns : pd.Index, optional, default = None

    Examples
    --------
    >>> from skpro.distributions.base._set import IntervalSet

    >>> s = IntervalSet(lower=0, upper=1, interval_type="[]")
    >>> s.contains(0.5)
    True
    >>> s.contains(-1)
    False
    """

    _tags = {
        "is_continuous": True,
    }

    _VALID_INTERVAL_TYPES = {"[]", "()", "[)", "(]"}

    def __init__(self, lower, upper, interval_type="[]", index=None, columns=None):
        self.lower = lower
        self.upper = upper
        self.interval_type = interval_type

        itype_arr = np.asarray(interval_type)
        if not np.all(np.isin(itype_arr, list(self._VALID_INTERVAL_TYPES))):
            raise ValueError(
                f"interval_type must be one of {self._VALID_INTERVAL_TYPES}, "
                f"but got '{interval_type}'."
            )

        super().__init__(index=index, columns=columns)
        if self.ndim > 0:
            self._get_bc_params(self.lower, self.upper)

    def contains(self, x):
        """Check whether ``x`` is contained in the interval.

        Parameters
        ----------
        x : scalar, np.ndarray, or pd.DataFrame
            Point(s) to check for membership.

        Returns
        -------
        bool, np.ndarray of bool, or pd.DataFrame of bool
            Membership mask.
        """
        x_is_df = isinstance(x, pd.DataFrame)
        if x_is_df:
            x_vals = x.values
        else:
            x_vals = np.asarray(x)

        if self.ndim > 0:
            lower, upper = self._get_bc_params(self.lower, self.upper)
        else:
            lower = np.asarray(self.lower, dtype=float)
            upper = np.asarray(self.upper, dtype=float)

        if self.ndim > 0 and x_vals.ndim == 1:
            x_vals = x_vals.reshape(-1, 1)

        itype = self.interval_type
        if itype == "[]":
            res = (x_vals >= lower) & (x_vals <= upper)
        elif itype == "()":
            res = (x_vals > lower) & (x_vals < upper)
        elif itype == "[)":
            res = (x_vals >= lower) & (x_vals < upper)
        elif itype == "(]":
            res = (x_vals > lower) & (x_vals <= upper)

        if isinstance(res, bool) or getattr(res, "ndim", 0) == 0:
            res = bool(res)
        return self._align_output(res, x)

    def boundary(self):
        """Return the boundary of the interval.

        Returns
        -------
        tuple of (lower, upper)
            The interval endpoints.
        """
        return self.lower, self.upper

    def __str__(self):
        """Return a human-readable string representation."""
        if not np.isscalar(self.lower) or not np.isscalar(self.upper):
            shape_str = f", shape={self.shape}" if self.ndim > 0 else ""
            return f"IntervalSet(tabular{shape_str})"

        lower = float(self.lower) if np.isscalar(self.lower) else self.lower
        upper = float(self.upper) if np.isscalar(self.upper) else self.upper
        return f"{self.interval_type[0]}{lower}, " f"{upper}{self.interval_type[1]}"

    def __repr__(self):
        """Return a developer-readable string representation."""
        return (
            f"IntervalSet(lower={self.lower!r}, upper={self.upper!r}, "
            f"interval_type={self.interval_type!r})"
        )


class FiniteSet(BaseSet):
    r"""Finite set of the real line.

    Represents a discrete set of the form :math:`\{a_1, \ldots, a_n\}`,
    where :math:`a_1, \ldots, a_n \in \mathbb{R}`.

    For tabular distributions, ``values`` is a flat list or 1D array of all
    possible support points. The same set of points applies to every entry
    of the distribution table — this is the common case for distributions
    like ``Poisson``, ``Binomial``, or ``Delta``.

    Parameters
    ----------
    values : list of int or float, or 1D array-like
        The elements contained in the set.
    index : pd.Index, optional, default = None
    columns : pd.Index, optional, default = None

    Examples
    --------
    >>> from skpro.distributions.base._set import FiniteSet

    >>> s = FiniteSet(values=[0, 1, 2])
    >>> s.contains(1)
    True
    >>> s.contains(4)
    False
    """

    _tags = {
        "is_discrete": True,
    }

    def __init__(self, values, index=None, columns=None):
        vals_arr = np.asarray(values)

        # Disambiguate global declarations
        # list like [1,2,3] is treated as global set(all row share same values)
        # [[1,2],[3,4]] is treated as tabular set (each row has different values)
        if vals_arr.dtype == object or vals_arr.ndim >= 2:
            self.values = vals_arr
            if vals_arr.ndim >= 2 and vals_arr.dtype != object:
                self.values = np.empty(len(vals_arr), dtype=object)
                for i in range(len(vals_arr)):
                    self.values[i] = list(vals_arr[i])
        else:
            self.values = np.asarray(values, dtype=float)

        super().__init__(index=index, columns=columns)

        if self.ndim > 0 and self.values.dtype == object:
            self._get_bc_params(self.values)

    def contains(self, x):
        """Check whether ``x`` is contained in the set.

        Parameters
        ----------
        x : scalar, np.ndarray, or pd.DataFrame
            Point(s) to check for membership.

        Returns
        -------
        bool, np.ndarray of bool, or pd.DataFrame of bool
            Membership mask.
        """
        x_is_df = isinstance(x, pd.DataFrame)
        x_vals = x.values if x_is_df else np.asarray(x)

        if self.ndim > 0 and x_vals.ndim == 1:
            x_vals = x_vals.reshape(-1, 1)

        if self.values.dtype == object:
            if self.ndim > 0:
                (values_bc,) = self._get_bc_params(self.values)
            else:
                values_bc = self.values

            def _in_set(val, allowed):
                if hasattr(allowed, "item"):
                    allowed = allowed.item()
                try:
                    return val in list(allowed)
                except TypeError:
                    return False

            vfunc = np.vectorize(_in_set, otypes=[bool])
            res = vfunc(x_vals, values_bc)
        else:
            res = np.isin(x_vals, self.values)

        if isinstance(res, bool) or getattr(res, "ndim", 0) == 0:
            res = bool(res)

        return self._align_output(res, x)

    def boundary(self):
        """Return the boundary of the set.

        Returns
        -------
        tuple of (lower, upper)
            The minimum and maximum limits of the discrete elements.
        """
        if self.values.dtype == object:
            if self.ndim > 0:
                (values_bc,) = self._get_bc_params(self.values)
            else:
                values_bc = self.values

            def _min_set(s):
                if hasattr(s, "item"):
                    s = s.item()
                lst = list(s)
                return float(np.min(lst)) if len(lst) > 0 else np.nan

            def _max_set(s):
                if hasattr(s, "item"):
                    s = s.item()
                lst = list(s)
                return float(np.max(lst)) if len(lst) > 0 else np.nan

            vfunc_min = np.vectorize(_min_set, otypes=[float])
            vfunc_max = np.vectorize(_max_set, otypes=[float])
            return vfunc_min(values_bc), vfunc_max(values_bc)
        else:
            if len(self.values) == 0:
                raise ValueError("FiniteSet is empty, so it has no boundaries.")
            return np.min(self.values), np.max(self.values)

    def __str__(self):
        """Return a human-readable string representation."""
        sorted_vals = sorted(self.values)
        return "{" + ", ".join(str(v) for v in sorted_vals) + "}"

    def __repr__(self):
        """Return a developer-readable string representation."""
        return f"FiniteSet(values={self.values.tolist()!r})"


class IntegerSet(BaseSet):
    r"""Set of consecutive integers.

    Represents a set of the form :math:`\{a, a+1, a+2, \ldots, b\}`,
    where :math:`a, b \in \mathbb{Z} \cup \{-\infty, +\infty\}`.

    This is the correct support representation for discrete distributions
    with integer-valued support, such as ``Poisson`` (``{0, 1, 2, ...}``),
    ``Binomial`` (``{0, 1, ..., n}``), or ``Geometric`` (``{0, 1, 2, ...}``).

    Unlike ``FiniteSet``, this class can represent **countably infinite** sets
    by setting ``upper=np.inf``. Unlike ``IntervalSet``, this class is tagged
    ``is_discrete=True`` and its ``contains`` method checks for integer values.

    Parameters
    ----------
    lower : int or array-like, optional, default = 0
        Lower bound of the integer range (inclusive).
    upper : int, float, or array-like, optional, default = np.inf
        Upper bound of the integer range (inclusive).
        Use ``np.inf`` for unbounded above.
    index : pd.Index, optional, default = None
    columns : pd.Index, optional, default = None

    Examples
    --------
    >>> from skpro.distributions.base._set import IntegerSet

    >>> s = IntegerSet(lower=0)  # {0, 1, 2, ...}
    >>> s.contains(3)
    True
    >>> s.contains(3.5)
    False
    """

    _tags = {
        "is_discrete": True,
    }

    def __init__(self, lower=0, upper=np.inf, index=None, columns=None):
        self.lower = lower
        self.upper = upper

        super().__init__(index=index, columns=columns)

        # Fail Fast: crash if parameter arrays don't broadcast to tabular shape
        if self.ndim > 0:
            self._get_bc_params(self.lower, self.upper)

    def contains(self, x):
        """Check whether ``x`` is contained in the integer set.

        Returns ``True`` for values that are integer-valued and
        within ``[lower, upper]``.

        Parameters
        ----------
        x : scalar, np.ndarray, or pd.DataFrame
            Point(s) to check for membership.

        Returns
        -------
        bool, np.ndarray of bool, or pd.DataFrame of bool
            Membership mask.
        """
        x_is_df = isinstance(x, pd.DataFrame)
        if x_is_df:
            x_vals = x.values
        else:
            x_vals = np.asarray(x, dtype=float)

        if self.ndim > 0:
            lower, upper = self._get_bc_params(self.lower, self.upper)
        else:
            lower = np.asarray(self.lower, dtype=float)
            upper = np.asarray(self.upper, dtype=float)

        if self.ndim > 0 and x_vals.ndim == 1:
            x_vals = x_vals.reshape(-1, 1)

        # check integer-valued: x == floor(x)
        is_int = np.equal(x_vals, np.floor(x_vals))
        in_range = (x_vals >= lower) & (x_vals <= upper)
        res = is_int & in_range

        if isinstance(res, bool) or getattr(res, "ndim", 0) == 0:
            res = bool(res)
        return self._align_output(res, x)

    def boundary(self):
        """Return the boundary of the integer set.

        Returns
        -------
        tuple of (lower, upper)
            The integer range endpoints.
        """
        return self.lower, self.upper

    def __str__(self):
        """Return a human-readable string representation."""
        if not np.isscalar(self.lower) or not np.isscalar(self.upper):
            shape_str = f", shape={self.shape}" if self.ndim > 0 else ""
            return f"IntegerSet(tabular{shape_str})"

        lower = int(self.lower) if np.isfinite(self.lower) else self.lower
        upper = int(self.upper) if np.isfinite(self.upper) else "∞"
        nxt = lower + 1 if np.isfinite(self.lower) else "..."
        return "{" + f"{lower}, {nxt}, ..., {upper}" + "}"

    def __repr__(self):
        """Return a developer-readable string representation."""
        return f"IntegerSet(lower={self.lower!r}, upper={self.upper!r})"


class UnionSet(BaseSet):
    r"""Union of multiple sets.

    Represents :math:`A_1 \\cup A_2 \\cup \\ldots \\cup A_n`.

    This is the key class for mixed distributions (e.g., ``ZeroInflated``),
    where the support is the union of a discrete mass point
    and a continuous interval.

    The ``contains`` method returns the logical **OR** of the children's
    ``contains`` results.

    Parameters
    ----------
    sets : positional args of BaseSet
        The component sets to take the union of.
    index : pd.Index, optional, default = None
    columns : pd.Index, optional, default = None

    Examples
    --------
    >>> from skpro.distributions.base._set import FiniteSet, IntervalSet, UnionSet

    >>> s = UnionSet(FiniteSet([0]), IntervalSet(1, 5, "()"))
    >>> s.contains(0)
    True
    >>> s.contains(3)
    True
    >>> s.contains(0.5)
    False
    """

    def __init__(self, *sets, index=None, columns=None):
        self.sets = sets
        super().__init__(index=index, columns=columns)

        # dynamic tag aggregation
        is_discrete = any(s.get_tag("is_discrete", raise_error=False) for s in sets)
        is_continuous = any(s.get_tag("is_continuous", raise_error=False) for s in sets)
        self.set_tags(**{"is_discrete": is_discrete, "is_continuous": is_continuous})

    def contains(self, x):
        """Check whether ``x`` is in the union (logical OR).

        Parameters
        ----------
        x : scalar, np.ndarray, or pd.DataFrame
            Point(s) to check for membership.

        Returns
        -------
        bool, np.ndarray of bool, or pd.DataFrame of bool
            Membership mask.
        """
        x_is_df = isinstance(x, pd.DataFrame)
        results = []
        for s in self.sets:
            if x_is_df and s.index is not None:
                # Align the heterogeneous dataframe boundaries before querying
                x_s = x.reindex(index=s.index, columns=s.columns)
                results.append(s.contains(x_s))
            else:
                results.append(s.contains(x))

        is_df = any(isinstance(r, pd.DataFrame) for r in results)
        is_series = any(isinstance(r, pd.Series) for r in results)

        if is_df or is_series:
            res = results[0]
            for i in range(1, len(results)):
                # Pandas automatically merges DataFrames handling alignments
                res = res | results[i]
        else:
            res = results[0]
            for i in range(1, len(results)):
                res = np.logical_or(res, results[i])

        return self._align_output(res, x)

    def boundary(self):
        """Return the union of boundaries.

        Returns
        -------
        tuple of (lower, upper)
            The global min and max of all component set boundaries.
        """
        lowers = []
        uppers = []
        for s in self.sets:
            try:
                l, u = s.boundary()
                if l is not None and u is not None:
                    lowers.append(np.asarray(l))
                    uppers.append(np.asarray(u))
            except (NotImplementedError, TypeError, ValueError):
                continue

        if not lowers:
            raise NotImplementedError("None of the component sets provide a boundary.")

        l_bc = np.broadcast_arrays(*lowers)
        u_bc = np.broadcast_arrays(*uppers)
        return np.min(l_bc, axis=0), np.max(u_bc, axis=0)

    def __str__(self):
        """Return a human-readable string representation."""
        return " ∪ ".join(str(s) for s in self.sets)

    def __repr__(self):
        """Return a developer-readable string representation."""
        sets_repr = ", ".join(repr(s) for s in self.sets)
        return f"UnionSet({sets_repr})"


class IntersectionSet(BaseSet):
    r"""Intersection of multiple sets.

    Represents :math:`A_1 \\cap A_2 \\cap \\ldots \\cap A_n`.

    This is the key class for truncated distributions, where the support
    is the intersection of the base distribution's support and the
    truncation interval. This avoids manual boundary recalculation.

    The ``contains`` method returns the logical **AND** of the children's
    ``contains`` results.

    Parameters
    ----------
    sets : positional args of BaseSet
        The component sets to take the intersection of.
    index : pd.Index, optional, default = None
    columns : pd.Index, optional, default = None

    Examples
    --------
    >>> from skpro.distributions.base._set import FiniteSet, IntersectionSet
    >>> from skpro.distributions.base._set import IntervalSet

    >>> base = FiniteSet([0, 1, 2, 3, 4, 5])
    >>> trunc = IntervalSet(2.5, 5.5, "[]")
    >>> s = IntersectionSet(base, trunc)
    >>> s.contains(3)
    True
    >>> s.contains(1)
    False
    """

    def __init__(self, *sets, index=None, columns=None):
        self.sets = sets
        super().__init__(index=index, columns=columns)

        # dynamic tag aggregation
        is_discrete = all(s.get_tag("is_discrete", raise_error=False) for s in sets)
        is_continuous = all(s.get_tag("is_continuous", raise_error=False) for s in sets)
        self.set_tags(**{"is_discrete": is_discrete, "is_continuous": is_continuous})

    def contains(self, x):
        """Check whether ``x`` is in the intersection (logical AND).

        Parameters
        ----------
        x : scalar, np.ndarray, or pd.DataFrame
            Point(s) to check for membership.

        Returns
        -------
        bool, np.ndarray of bool, or pd.DataFrame of bool
            Membership mask.
        """
        x_is_df = isinstance(x, pd.DataFrame)
        results = []
        for s in self.sets:
            if x_is_df and s.index is not None:
                x_s = x.reindex(index=s.index, columns=s.columns)
                results.append(s.contains(x_s))
            else:
                results.append(s.contains(x))

        is_df = any(isinstance(r, pd.DataFrame) for r in results)
        is_series = any(isinstance(r, pd.Series) for r in results)

        if is_df or is_series:
            res = results[0]
            for i in range(1, len(results)):
                res = res & results[i]
        else:
            res = results[0]
            for i in range(1, len(results)):
                res = np.logical_and(res, results[i])

        return self._align_output(res, x)

    def boundary(self):
        """Return the intersection of boundaries.

        Returns
        -------
        tuple of (lower, upper)
            The intersection of all component set boundaries.
        """
        lowers = []
        uppers = []
        for s in self.sets:
            try:
                l, u = s.boundary()
                if l is not None and u is not None:
                    lowers.append(np.asarray(l))
                    uppers.append(np.asarray(u))
            except (NotImplementedError, TypeError, ValueError):
                continue

        if not lowers:
            raise NotImplementedError("None of the component sets provide a boundary.")

        l_bc = np.broadcast_arrays(*lowers)
        u_bc = np.broadcast_arrays(*uppers)

        out_lower = np.max(l_bc, axis=0)
        out_upper = np.min(u_bc, axis=0)

        # Discard invalid overlapping segments (Null Intersections)
        empty_mask = out_lower > out_upper
        out_lower = np.where(empty_mask, np.nan, out_lower)
        out_upper = np.where(empty_mask, np.nan, out_upper)

        if not empty_mask.shape and empty_mask:
            out_lower, out_upper = np.nan, np.nan

        return out_lower, out_upper

    def __str__(self):
        """Return a human-readable string representation."""
        return " ∩ ".join(str(s) for s in self.sets)

    def __repr__(self):
        """Return a developer-readable string representation."""
        sets_repr = ", ".join(repr(s) for s in self.sets)
        return f"IntersectionSet({sets_repr})"


class EmptySet(BaseSet):
    """The empty set.

    ``contains`` always returns ``False``.

    Parameters
    ----------
    index : pd.Index, optional, default = None
    columns : pd.Index, optional, default = None
    """

    def __init__(self, index=None, columns=None):
        super().__init__(index=index, columns=columns)

    def contains(self, x):
        """Check whether ``x`` is in the empty set (always False).

        Parameters
        ----------
        x : scalar, np.ndarray, or pd.DataFrame
            Point(s) to check for membership.

        Returns
        -------
        bool, np.ndarray of bool, or pd.DataFrame of bool
            Always False.
        """
        if hasattr(x, "shape") and getattr(x, "ndim", 0) > 0:
            res = np.zeros(x.shape, dtype=bool)
        else:
            res = False
        return self._align_output(res, x)

    def boundary(self):
        """Return the boundary (empty)."""
        return np.array([])

    def __str__(self):
        """Return a human-readable string representation."""
        return "∅"

    def __repr__(self):
        """Return a developer-readable string representation."""
        return "EmptySet()"


class RealSet(BaseSet):
    r"""The set of all real numbers :math:`\mathbb{R}`.

    ``contains`` always returns ``True``.
    This is the default support for unbounded continuous distributions
    like ``Normal`` or ``Cauchy``.

    Parameters
    ----------
    index : pd.Index, optional, default = None
    columns : pd.Index, optional, default = None
    """

    _tags = {
        "is_continuous": True,
    }

    def __init__(self, index=None, columns=None):
        super().__init__(index=index, columns=columns)

    def contains(self, x):
        r"""Check whether ``x`` is in :math:`\mathbb{R}` (always True).

        Parameters
        ----------
        x : scalar, np.ndarray, or pd.DataFrame
            Point(s) to check for membership.

        Returns
        -------
        bool, np.ndarray of bool, or pd.DataFrame of bool
            Always True.
        """
        if hasattr(x, "shape") and getattr(x, "ndim", 0) > 0:
            res = np.ones(x.shape, dtype=bool)
        else:
            res = True
        return self._align_output(res, x)

    def boundary(self):
        """Return the boundary (-inf, +inf)."""
        return -np.inf, np.inf

    def __str__(self):
        """Return a human-readable string representation."""
        return "ℝ"

    def __repr__(self):
        """Return a developer-readable string representation."""
        return "RealSet()"
