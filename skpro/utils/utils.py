# LEGACY MODULE - TODO: remove or refactor
"""Utilities for skpro."""


def not_existing(f):
    """Decorate an interface method to declare it theoretically non-existent.

    Parameters
    ----------
    f : callable
        Method to decorate.

    Returns
    -------
    f : callable
        Decorated method.
    """
    f.not_existing = True

    return f


def ensure_existence(f):
    """Ensure that method is not marked as non-existent.

    Parameters
    ----------
    f : callable
        Method to check.

    Raises
    ------
    NotImplementedError
        If the method is marked as non-existent.

    Returns
    -------
    f : callable
        Method f.
    """
    if getattr(f, "not_existing", False):
        raise NotImplementedError(
            "The distribution has no " + f.__name__ + " function. "
            "You may use an adapter that supports its approximation."
        )

    return f


def to_percent(value, return_float=True):
    """Convert values into a percent representation.

    Parameters
    ----------
    value : int or float
        Number representing a percentage.
    return_float : bool, optional (default=True)
        If true, float representing the percentage is returned.

    Returns
    -------
    int or float
        A percentage.
    """

    def percent(p):
        if return_float:
            return float(p)
        else:
            return int(round(p * 100))

    if isinstance(value, int):
        value = float(value) / 100.0

    if value <= 0:
        return percent(0)
    else:
        return percent(value)
