"""Common utilities for adapters."""

import numpy as np


def _clip_surv(surv_arr):
    """Clips improper survival function values to proper range.

    Enforces: values are in [0, 1] and are monotonically decreasing.

    First clips to [0, 1], then enforces monotonicity, by replacing
    any value with minimum of itself and any previous values.

    Parameters
    ----------
    surv_arr : 2D np.ndarray
        Survival function values.
        index 0 is instance index.
        index 1 is time index, increasing.

    Returns
    -------
    surv_arr_clipped : 2D np.ndarray
        Clipped survival function values.
    surv_arr_diff : 2D np.ndarray
        Difference of clipped survival function values.
        Same as np.diff(surv_arr_clipped, axis=1, prepend=1).
        Returned to avoid recomputation, if needed later in context.
    clipped : boolean
        Whether clipping was needed.
    """
    too_large = surv_arr > 1
    too_small = surv_arr < 0

    surv_arr[too_large] = 1
    surv_arr[too_small] = 0

    surv_arr_diff = _surv_diff(surv_arr)

    # avoid iterative minimization if no further clipping is needed
    if not (surv_arr_diff > 0).any():
        clipped = too_large.any() or too_small.any()
        return surv_arr, surv_arr_diff, clipped

    # enforce monotonicity
    # iterating from left to right ensures values are replaced
    # with minimum of itself and all values to the left
    for i in range(1, surv_arr.shape[1]):
        surv_arr[:, i] = np.minimum(surv_arr[:, i], surv_arr[:, i - 1])

    surv_arr_diff = _surv_diff(surv_arr)

    return surv_arr, surv_arr_diff, True


def _surv_diff(surv_arr):
    """Compute difference of survival function values.

    Parameters
    ----------
    surv_arr : 2D np.ndarray
        Survival function values.
        index 0 is instance index.
        index 1 is time index, increasing.

    Returns
    -------
    surv_arr_diff : 2D np.ndarray, same shape as surv_arr
        Difference of survival function values.
        Same as np.diff(surv_arr, axis=1, prepend=1, append=0),
        then summing the last two columns to become one column
    """
    surv_arr_diff = np.diff(surv_arr, axis=1, prepend=1, append=0)

    surv_arr_diff[:, -2] = surv_arr_diff[:, -2] + surv_arr_diff[:, -1]
    surv_arr_diff = surv_arr_diff[:, :-1]

    return surv_arr_diff
