"""Utility functions for numpy/sklearn related matters."""

__authors__ = ["fkiraly"]


def flatten_to_1D_if_colvector(y):
    """Flattens a numpy array to 1D if it is a 2D column vector.

    Parameters
    ----------
    y : numpy array, 1D or 2D
        Array to flatten

    Returns
    -------
    y_flat : numpy array
        1D flattened array if y was 2D column vector, or 1D already
        otherwise, returne y unchanged
    """
    if len(y.shape) == 2 and y.shape[1] == 1:
        y_flat = y.flatten()
    else:
        y_flat = y

    return y_flat
