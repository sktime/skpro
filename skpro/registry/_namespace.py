"""Unique namespace registry."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# based on the sktime utility of the same name

__all__ = ["_namespace"]


def _namespace(include_deps=False):
    """Return the unique namespace registry.

    Parameters
    ----------
    include_deps : bool, optional (default=False)
        Whether to include dependent namespaces.

        * If False, returns namespace of ``skpro`` only.
        * If True, includes the following dependent namespaces:

            * ``scikit-learn``

    Returns
    -------
    namespace_registry : dict
        Dictionary of the combined namespace.

        Contains pointers to classes and functions from the respective namespaces.
        The keys are names, and the values are pointers.
    """
    from skpro.registry._lookup import all_objects
    from skpro.registry._lookup_sklearn import _all_sklearn_estimators

    # retrieve all estimators from skpro and sklearn for namespace resolution
    namespace_dict_skpro = dict(all_objects())  # noqa: F841

    if include_deps:
        namespace_dict_sklearn = dict(_all_sklearn_estimators())  # noqa: F841
        namespace_dict_sklearn.update(namespace_dict_skpro)
        namespace_dict = namespace_dict_sklearn
    else:
        namespace_dict = namespace_dict_skpro

    return namespace_dict
