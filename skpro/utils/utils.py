# LEGACY MODULE - TODO: remove or refactor


def not_existing(f):
    """
    Decorates an interface method to declare it theoretically non existent

    Parameters
    ----------
    f   Method to decorate

    Returns
    -------
    Decorated method
    """
    f.not_existing = True

    return f


def ensure_existence(f):
    """Ensures that method is not marked as non_existent

    Parameters
    ----------
    f  Method

    Raises
    ------
    NotImplementedError if the method is marked as non existent

    Returns
    -------
    Method f
    """
    if getattr(f, "not_existing", False):
        raise NotImplementedError(
            "The distribution has no " + f.__name__ + " function. "
            "You may use an adapter that supports its approximation."
        )

    return f


def to_percent(value, return_float=True):
    """Converts values into a percent representation

    Args:
        value: int/float
            Number representing a percentage
        return_float: bool
            If true, float representing the percentage is returned

    Returns: int/float
        A percentage
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
