
import math

import numpy as np
from numba import jit


@jit(nopython=True)
def le(z, z_searched, inclusive):
    """
    Binary search for the **last** element **less than or equal to**
    a given value in a sorted float64 array

    If there are issues with float64 precision, please **add** a suitable
    ``epsilon`` to ``z_searched`` first.

    :param z: sorted array to search in; consecutive repetitions of values are
        allowed.
    :type z: :class:`numpy.ndarray` `(float64, ndim=1)`

    :param z_searched: value to look for
    :type z_searched: float64

    :param inclusive: whether to always return a valid index, see below
    :type inclusive: bool

    :return: the index of the last element **less than or equal to**
        ``z_searched``; If ``z_searched`` is less than all elements in ``z``,
        -1 or 0 is returned for ``inclusive=False, True``, respectively.
    :rtype: int64_t
    """
    i_left = 0 if inclusive else -1
    i_right = z.shape[0]
    while i_left < i_right - 1:
        i = (i_left + i_right) // 2
        if z[i] <= z_searched:
            i_left = i
        else:
            i_right = i
    return i_left


@jit(nopython=True)
def ge(z, z_searched, inclusive):
    """
    Binary search for the **first** element **greater than or equal to**
    a given value in a sorted float64 array

    This function implements the same functionality as the C++ function
    ``std::lower_bound``. C++'s ``std::equal_range`` corresponds to the tuple
    :func:`ge_lim`, :func:`gt_lim`.

    If there are issues with float64 precision, please **subtract** a suitable
    ``epsilon`` from ``z_searched`` first.

    :param z: sorted array to search in; consecutive repetitions of values are
        allowed.
    :type z: :class:`numpy.ndarray` `(float64, ndim=1)`

    :param z_searched: value to look for
    :type z_searched: float64

    :param inclusive: whether to always return a valid index, see below
    :type inclusive: bool

    :return: the index of the first element **greater than or equal to**
        ``z_searched``; If ``z_searched`` is greater than all elements in ``z``,
        ``len(z)`` or ``len(z) - 1`` is returned for
        ``inclusive=False, True``, respectively.
    :rtype: int64_t
    """
    n = z.shape[0]
    i_left = -1
    i_right = n - 1 if inclusive else n
    while i_left < i_right - 1:
        i = (i_left + i_right) // 2
        if z[i] >= z_searched:
            i_right = i
        else:
            i_left = i
    return i_right


@jit(nopython=True)
def eq_multi(z, z_searched, u, epsilon, result):
    """
    Search the values of `z_searched` in z and return u[i_found] if
    equality of z_searched[i_searched] and z[i_found] is given. Otherwise
    `numpy.nan` is returned.

    :param z: sorted array to search in
    :type z: :class:`numpy.ndarray` (float64, ndim=1)

    :param z_searched: Values to search for.
    :type z_searched: :class:`numpy.ndarray` (float64, ndim=1)

    :param u: Array from which to take the result. It is in one-to-one
        correspondence to z.
    :type u: :class:`numpy.ndarray` `(float64, ndim=1)`

    :param epsilon: In order to avoid issues with float64 precision, this
        epsilon is used.
    :type epsilon: float64

    :param result: where to put the results; passing the same array as
        ``z_searched`` is supported.
    :type result: :class:`numpy.ndarray` (float64, ndim=1), same length as
        ``z_searched``
    """
    n = z_searched.shape[0]

    if result.shape[0] != n:
        raise ValueError("Inconsistent lengths of z_searched and result.")
    if z.shape[0] != u.shape[0]:
        raise ValueError("Inconsistent lengths of z and u.")
    for i in range(n):
        ind = le(z, z_searched[i] + epsilon, 0)
        if check_equal(z[ind], z_searched[i], epsilon, 0.0):
            result[i] = u[ind]
        else:
            result[i] = np.nan


@jit(nopython=True)
def ge_multi(z, z_searched, inclusive, result):
    """
    Binary search for the **first** elements **greater than or equal to**
    given values in a sorted float64 array

    For each value in ``z_searched``, :func:`ge` is called.

    If there are issues with float64 precision, please **subtract** a suitable
    ``epsilon`` from ``z_searched`` first.

    :param z: sorted array to search in; consecutive repetitions of values are
        allowed.
    :type z: :class:`numpy.ndarray` `(float64, ndim=1)`

    :param z_searched: value to look for
    :type z_searched: float64

    :param inclusive: whether to always return a valid index, see :func:`ge`
    :type inclusive: bool

    :param result: where to put the results
    :type result: :class:`numpy.ndarray` (float64, ndim=1), same length as
        ``z_searched``
    """
    n = z_searched.shape[0]

    if result.shape[0] != n:
        raise ValueError("Inconsistent lengths of z_searched and result.")
    for i in range(n):
        result[i] = ge(z, z_searched[i], inclusive)


@jit(nopython=True)
def le_interp(z, z_searched, u, out_left, epsilon):
    """
    Interpolation of a value between the position found by :func:`le` (i.e. the
    **last** element **less than or equal to** a given value in a sorted float64
    array) and the next position

    :param z: sorted array to search in
    :type z: :class:`numpy.ndarray` (float64, ndim=1)

    :param z_searched: Value to search for.
    :type z_searched: float64

    :param u: Array from which to take the result. It is in one-to-one
        correspondence to z.
    :type u: :class:`numpy.ndarray` `(float64, ndim=1)`

    :param out_left: value to return for ``z_searched < z[0]``.
    :type out_left: float64

    :param epsilon: In order to avoid issues with float64 precision, this
        epsilon is added to ``z_searched`` for the binary search, but not in
        the interpolation.
    :type epsilon: float64

    :return: If ``z_searched`` is found in z, the corresponding value ``u[i]``.
        Otherwise a value between ``u[i]`` and ``u[i + 1]`` interpolated in
        accord with the corresponding z values.

        If ``z_searched`` is not a finite value or ``z`` is empty, ``np.nan``
        is returned.
    :rtype: float64
    """
    n = z.shape[0]

    if not math.isfinite(z_searched) or n == 0:
        return np.nan

    i = le(z, z_searched + epsilon, 0)
    z_found = z[i]
    u0 = u[i]

    if i < 0:
        return out_left

    if i < n - 1 and z_found < z_searched:
        u0 += (z_searched - z_found) / (z[i + 1] - z_found) * (u[i + 1] - u0)

    return u0


@jit(nopython=True)
def le_interp_multi(z, z_searched, u, out_left, epsilon, result):
    """
    Interpolation of values between the position found by :func:`le` (i.e. the
    **last** element **less than or equal to** a given value in a sorted float64
    array) and the next position

    For each element in ``z_searched``, :func:`le_interp`
    is called and the result is put in the corresponding slot in ``result``.

    :param z: sorted array to search in
    :type z: :class:`numpy.ndarray` (float64, ndim=1)

    :param z_searched: Values to search for.
    :type z_searched: :class:`numpy.ndarray` (float64, ndim=1)

    :param u: Array from which to take the result. It is in one-to-one
        correspondence to z.
    :type u: :class:`numpy.ndarray` `(float64, ndim=1)`

    :param out_left: value to return for ``z_searched < z[0]``.
    :type out_left: float64

    :param epsilon: In order to avoid issues with float64 precision, this
        epsilon is added to ``z_searched`` for the binary search, but not in
        the interpolation.
    :type epsilon: float64

    :param result: where to put the results; passing the same array as
        ``z_searched`` is supported.
    :type result: :class:`numpy.ndarray` (float64, ndim=1), same length as
        ``z_searched``
    """
    n = z_searched.shape[0]
    if result.shape[0] != n:
        raise ValueError("Inconsistent lengths of z_searched and result.")
    for i in range(n):
        result[i] = le_interp(z, z_searched[i], u, out_left, epsilon)


@jit(nopython=True)
def ge_lim(z, z_searched, inclusive, i_left, i_right):
    """
    Binary search for the **first** element **greater than or equal to**
    a given value in a sorted float64 array

    This function implements the same functionality as the C++ function
    ``std::lower_bound``. C++'s ``std::equal_range`` corresponds to the tuple
    :func:`ge_lim`, :func:`gt_lim`.

    If there are issues with float64 precision, please **subtract** a suitable
    ``epsilon`` from ``z_searched`` first.

    :param z: sorted array to search in; consecutive repetitions of values are
        allowed.
    :type z: :class:`numpy.ndarray` `(float64, ndim=1)`

    :param z_searched: value to look for
    :type z_searched: float64

    :param inclusive: whether to always return a valid index, see below
    :type inclusive: bool

    :param i_left: The range of array indices considered is
        ``[i_left, i_right)``.
    :type i_left: int64_t

    :param i_right: see ``i_left``.
    :type i_right: int64_t

    :return: the index of the first element **greater than or equal to**
        ``z_searched``; If ``z_searched`` is greater than all elements in ``z``,
        ``i_right`` or ``i_right - 1`` is returned for
        ``inclusive=False, True``, respectively.
    :rtype: int64_t
    """
    if inclusive:
        i_right -= 1
    i_left -= 1

    while i_left < i_right - 1:
        i = (i_left + i_right) // 2
        if z[i] >= z_searched:
            i_right = i
        else:
            i_left = i
    return i_right


@jit(nopython=True)
def check_equal(z_1, z_2, atol, rtol):
    r"""
    Check if two doubles are equal within the specified relative and absolute
    precision. This function is similar to :func:`numpy.allclose` with the
    difference that it is symmetric in `z1` and `z2` regarding relative
    comparisons.

    By default NaN's are considered unequal.

    .. math::

        | z_1 - z_2 | \le a_{tol} + r_{tol} \cdot (|z_1| + |z_2|)

    :param z_1: first input double
    :type z_1: double

    :param z_2: second input double
    :type z_2: double

    :param atol: absolute tolerance for comparisons:
        :math:`| z_1 - z_2 | \le a_{tol}`
    :type atol: double

    :param rtol: relative tolerance for comparisons:
        :math:`| z_1 - z_2 | \le r_{tol} \cdot (|z_1| + |z_2|)`
    :type rtol: double

    :rtype: bool
    """
    sum_abs = math.fabs(z_1) + math.fabs(z_2)
    abs_diff = math.fabs(z_1 - z_2)
    return abs_diff <= atol + (rtol * sum_abs)
