

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
    """ Ensures that method is not marked as non_existent

    Parameters
    ----------
    f  Method

    Raises
    ------
    NotImplementedError if the method is marked as non existent

    Returns
    -------
    function f
    """
    if getattr(f, 'not_existing', False):
        raise NotImplementedError('The distribution has no ' + f.__name__ + ' function. '
                                  'You may use an adapter that supports its approximation.')

    return f
