

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