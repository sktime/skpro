"""Register of estimator base classes corresponding to skpro scitypes.

To add a new base class to the register,
define a new class inheriting from ``_BaseScitypeOfObject``, fill in the tags below,
and implement the methods below.

Tags to fill in:

* ``scitype_name`` : scitype shorthand string. IMPORTANT: this will be used
  across the codebase as a unique identifier.
* ``short_descr`` : short English description of the scitype
* ``parent_scitype`` : parent scitype shorthand string, for scitype inheritance.
  IF not filled in, will inherit from ``object`` scitype.
* ``mixin`` : whether this is a mixin scitype (True) or full scitype (False).
  Only fill in with value ``True`` if used as a mixin class.

Class methods to implement:

* ``get_base_class`` : should return the base class corresponding to the scitype.
  The base class should inherit from ``skpro.base.BaseObject``, or a subclass thereof.
* ``get_test_class`` : should return the test class for the scitype.
  This class should follow the pattern of ``TestAll[ScitypeName]s`` classes in
  ``skpro``.

For examples, see below, and follow the pattern to add new scitypes.
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# based on the sktime module of same name

import inspect
import sys
from functools import lru_cache

from skbase.base import BaseObject


class _BaseScitypeOfObject(BaseObject):
    """Base class for all object scitypes."""

    _tags = {
        "object_type": "scitype:object",
        "scitype_name": "fill_this_in",
        "parent_scitype": None,
        "short_descr": "describe the scitype here",
        "mixin": False,
    }

    @classmethod
    def get_test_class(cls):
        """Return test class for the scitype."""
        return None


# ----------------------------------
# Core scitypes
# ----------------------------------


class object(_BaseScitypeOfObject):
    """Universal type for all objects."""

    _tags = {
        "scitype_name": "object",
        "short_descr": "base scitype for all objects",
    }

    @classmethod
    def get_base_class(cls):
        """Return base class for object scitype."""
        from skpro.base import BaseObject

        return BaseObject

    @classmethod
    def get_test_class(cls):
        """Return test class for object scitype."""
        from skpro.tests.test_all_estimators import TestAllObjects

        return TestAllObjects


class estimator(_BaseScitypeOfObject):
    """Estimator objects, i.e., objects with fit method."""

    _tags = {
        "scitype_name": "estimator",
        "short_descr": "estimator = object with fit",
        "parent_scitype": "object",
    }

    @classmethod
    def get_base_class(cls):
        """Return base class for estimator scitype."""
        from skpro.base import BaseEstimator

        return BaseEstimator

    @classmethod
    def get_test_class(cls):
        """Return test class for estimator scitype."""
        from skpro.tests.test_all_estimators import TestAllEstimators

        return TestAllEstimators


# ----------------------------------
# Probabilistic regression scitypes
# ----------------------------------


class regressor_proba(_BaseScitypeOfObject):
    """Probabilistic regressor."""

    _tags = {
        "scitype_name": "regressor_proba",
        "short_descr": "probabilistic regressor",
        "parent_scitype": "estimator",
    }

    @classmethod
    def get_base_class(cls):
        """Return base class for regressor_proba scitype."""
        from skpro.regression.base import BaseProbaRegressor

        return BaseProbaRegressor

    @classmethod
    def get_test_class(cls):
        """Return test class for regressor_proba scitype."""
        from skpro.regression.tests.test_all_regressors import TestAllRegressors

        return TestAllRegressors


# ----------------------------------
# Distribution scitypes
# ----------------------------------


class distribution(_BaseScitypeOfObject):
    """Probability distribution."""

    _tags = {
        "scitype_name": "distribution",
        "short_descr": "probability distribution",
        "parent_scitype": "object",
    }

    @classmethod
    def get_base_class(cls):
        """Return base class for distribution scitype."""
        from skpro.distributions.base import BaseDistribution

        return BaseDistribution

    @classmethod
    def get_test_class(cls):
        """Return test class for distribution scitype."""
        from skpro.distributions.tests.test_all_distrs import TestAllDistributions

        return TestAllDistributions


# ----------------------------------
# Metric scitypes
# ----------------------------------


class metric(_BaseScitypeOfObject):
    """Performance metric for probabilistic predictions."""

    _tags = {
        "scitype_name": "metric",
        "short_descr": "performance metric",
        "parent_scitype": "object",
    }

    @classmethod
    def get_base_class(cls):
        """Return base class for metric scitype."""
        from skpro.metrics.base import BaseProbaMetric

        return BaseProbaMetric


class metric_distr(_BaseScitypeOfObject):
    """Performance metric for distribution predictions."""

    _tags = {
        "scitype_name": "metric_distr",
        "short_descr": "performance metric for distribution predictions",
        "parent_scitype": "metric",
    }

    @classmethod
    def get_base_class(cls):
        """Return base class for metric_distr scitype."""
        from skpro.metrics.base import BaseDistrMetric

        return BaseDistrMetric

    @classmethod
    def get_test_class(cls):
        """Return test class for metric_distr scitype."""
        from skpro.metrics.tests.test_distr_metrics import TestAllDistrMetrics

        return TestAllDistrMetrics


# ----------------------------------
# Datatype scitypes
# ----------------------------------


class datatype(_BaseScitypeOfObject):
    """Datatype specification."""

    _tags = {
        "scitype_name": "datatype",
        "short_descr": "datatype specification",
        "parent_scitype": "object",
    }

    @classmethod
    def get_base_class(cls):
        """Return base class for datatype scitype."""
        from skpro.datatypes._base import BaseDatatype

        return BaseDatatype


class converter(_BaseScitypeOfObject):
    """Datatype converter."""

    _tags = {
        "scitype_name": "converter",
        "short_descr": "datatype converter",
        "parent_scitype": "object",
    }

    @classmethod
    def get_base_class(cls):
        """Return base class for converter scitype."""
        from skpro.datatypes._base import BaseConverter

        return BaseConverter


class datatype_example(_BaseScitypeOfObject):
    """Datatype example fixture."""

    _tags = {
        "scitype_name": "datatype_example",
        "short_descr": "datatype example fixture",
        "parent_scitype": "object",
    }

    @classmethod
    def get_base_class(cls):
        """Return base class for datatype_example scitype."""
        from skpro.datatypes._base import BaseExample

        return BaseExample


# ----------------------------------
# utility functions for base classes
# ----------------------------------


@lru_cache
def _get_base_classes(mixin=False):
    """Get all object scitype classes in this module.

    Parameters
    ----------
    mixin : bool, default=False
        If True, return only mixin scitypes.
        If False, return only non-mixin scitypes.

    Returns
    -------
    clss : tuple of classes
        All scitype classes in this module matching the mixin parameter.
    """
    clss = inspect.getmembers(sys.modules[__name__], inspect.isclass)

    base_cls = _BaseScitypeOfObject
    base_cls_name = base_cls.__name__

    def is_base_class(cl):
        return cl.__name__ != base_cls_name and issubclass(cl, base_cls)

    clss = [cl for _, cl in clss if is_base_class(cl)]
    clss = [cl for cl in clss if cl.get_class_tags().get("mixin", False) == mixin]
    clss = tuple(clss)
    return clss


def _construct_child_tree(mode="class"):
    """Construct inheritance tree for all scitypes.

    Parameters
    ----------
    mode : str, one of "class" or "str", default="class"
        If "class", keys and values are scitype classes.
        If "str", keys and values are scitype name strings.

    Returns
    -------
    child_tree : dict
        Dictionary mapping each scitype to its list of child scitypes.
    """
    return _construct_child_tree_cached(mode=mode).copy()


@lru_cache
def _construct_child_tree_cached(mode="class"):
    """Construct inheritance tree for all scitypes, cached version."""
    clss = _get_base_classes()

    def _entry_for(cl):
        if mode == "class":
            return cl
        elif mode == "str":
            return cl.get_class_tags()["scitype_name"]

    child_tree = {_entry_for(cl): [] for cl in clss}
    for cl in clss:
        parent_scitype = cl.get_class_tags()["parent_scitype"]
        if parent_scitype is not None:
            if parent_scitype not in child_tree:
                child_tree[parent_scitype] = []
            child_tree[parent_scitype].append(_entry_for(cl))

    return child_tree


def _get_all_descendants(scitype):
    """Get all descendants of a given scitype.

    Parameters
    ----------
    scitype : class or str
        Scitype class or scitype name string.

    Returns
    -------
    descendants : list
        List of all descendant scitypes (including the input scitype).
    """
    return _get_all_descendants_cached(scitype).copy()


@lru_cache
def _get_all_descendants_cached(scitype):
    """Get all descendants of a given scitype, cached version."""
    if isinstance(scitype, str):
        mode = "str"
    else:
        mode = "class"

    child_tree = _construct_child_tree(mode=mode)
    children = child_tree.get(scitype, [])
    if len(children) == 0:
        return [scitype]

    descendants = [x for child in children for x in _get_all_descendants(child)]
    descendants += [scitype]
    descendants = sorted(descendants, key=str)
    return descendants.copy()


@lru_cache
def _construct_base_class_register(mixin=False):
    """Generate the register from the classes in this module.

    Parameters
    ----------
    mixin : bool, default=False
        If True, include only mixin scitypes.
        If False, include only non-mixin scitypes.

    Returns
    -------
    register : list of tuples
        Each tuple contains (scitype_name, base_class, short_descr).
    """
    clss = _get_base_classes(mixin=mixin)

    register = []
    for cl in clss:
        cl_tags = cl.get_class_tags()

        scitype_name = cl_tags["scitype_name"]
        short_descr = cl_tags["short_descr"]
        base_cls_ref = cl.get_base_class()

        register.append((scitype_name, base_cls_ref, short_descr))

    return register


def get_base_class_for_str(scitype_str):
    """Return base class for a given scitype string.

    Parameters
    ----------
    scitype_str : str or list of str
        Scitype name string(s).

    Returns
    -------
    base_class : class or list of classes
        Base class(es) corresponding to the scitype string(s).

    Raises
    ------
    KeyError
        If scitype_str is not a valid scitype name.

    Examples
    --------
    >>> from skpro.registry._base_classes import get_base_class_for_str
    >>> get_base_class_for_str("regressor_proba")  # doctest: +SKIP
    <class 'skpro.regression.base._base.BaseProbaRegressor'>
    """
    if isinstance(scitype_str, list):
        return [get_base_class_for_str(s) for s in scitype_str]

    base_classes = _get_base_classes()
    base_classes += _get_base_classes(mixin=True)
    base_class_lookup = {cl.get_class_tags()["scitype_name"]: cl for cl in base_classes}
    base_cls = base_class_lookup[scitype_str].get_base_class()
    return base_cls


def get_test_class_for_str(scitype_str):
    """Return test class for a given scitype string.

    Parameters
    ----------
    scitype_str : str or list of str
        Scitype name string(s).

    Returns
    -------
    test_class : class, None, or list
        Test class(es) corresponding to the scitype string(s).
        Returns None if no test class is defined for the scitype.

    Raises
    ------
    KeyError
        If scitype_str is not a valid scitype name.

    Examples
    --------
    >>> from skpro.registry._base_classes import get_test_class_for_str
    >>> get_test_class_for_str("regressor_proba")  # doctest: +SKIP
    <class 'skpro.regression.tests.test_all_regressors.TestAllRegressors'>
    """
    if isinstance(scitype_str, list):
        return [get_test_class_for_str(s) for s in scitype_str]

    base_classes = _get_base_classes()
    base_classes += _get_base_classes(mixin=True)
    base_class_lookup = {cl.get_class_tags()["scitype_name"]: cl for cl in base_classes}
    test_cls = base_class_lookup[scitype_str].get_test_class()
    return test_cls


def get_base_class_register(mixin=False, include_baseobjs=True):
    """Return register of object scitypes and base classes in skpro.

    Parameters
    ----------
    mixin : bool, default=False
        If True, include only mixin scitypes.
        If False, include only non-mixin scitypes.
    include_baseobjs : bool, default=True
        If True, include "object" and "estimator" base scitypes.
        If False, exclude them.

    Returns
    -------
    register : list of tuples
        Each tuple contains (scitype_name, base_class, short_descr).

    Examples
    --------
    >>> from skpro.registry._base_classes import get_base_class_register
    >>> register = get_base_class_register()  # doctest: +SKIP
    """
    raw_list = list(_construct_base_class_register(mixin=mixin))

    if not include_baseobjs:
        raw_list = [x for x in raw_list if x[0] not in ["object", "estimator"]]

    distr = [x for x in raw_list if x[0] == "distribution"]
    rest = [x for x in raw_list if x[0] != "distribution"]
    reordered_list = rest + distr

    return reordered_list.copy()


@lru_cache
def _construct_scitype_list(mixin=False):
    """Generate list of scitype strings from the register."""
    clss = _get_base_classes(mixin=mixin)

    scitype_list = []
    for cl in clss:
        tags = cl.get_class_tags()
        scitype_list.append((tags["scitype_name"], tags["short_descr"]))
    return scitype_list


def get_obj_scitype_list(mixin=False, include_baseobjs=True, return_descriptions=False):
    """Return list of object scitype shorthands in skpro.

    Parameters
    ----------
    mixin : bool, default=False
        If True, include only mixin scitypes.
        If False, include only non-mixin scitypes.
    include_baseobjs : bool, default=True
        If True, include "object" and "estimator" base scitypes.
        If False, exclude them.
    return_descriptions : bool, default=False
        If True, return tuples of (scitype_name, short_descr).
        If False, return only scitype_name strings.

    Returns
    -------
    scitype_list : list
        List of scitype name strings, or tuples if return_descriptions=True.

    Examples
    --------
    >>> from skpro.registry._base_classes import get_obj_scitype_list
    >>> get_obj_scitype_list()  # doctest: +SKIP
    ['converter', 'datatype', 'datatype_example', 'estimator', 'metric', ...]
    """
    raw_list = list(_construct_scitype_list(mixin=mixin))

    if not include_baseobjs:
        raw_list = [x for x in raw_list if x[0] not in ["object", "estimator"]]

    distr = [x for x in raw_list if x[0] == "distribution"]
    rest = [x for x in raw_list if x[0] != "distribution"]
    reordered_list = rest + distr

    if return_descriptions:
        return reordered_list.copy()
    else:
        return [x[0] for x in reordered_list].copy()


def get_base_class_list(mixin=False, include_baseobjs=True):
    """Return list of base classes in skpro.

    Parameters
    ----------
    mixin : bool, default=False
        If True, include only mixin scitypes.
        If False, include only non-mixin scitypes.
    include_baseobjs : bool, default=True
        If True, include "object" and "estimator" base scitypes.
        If False, exclude them.

    Returns
    -------
    base_class_list : list of classes
        List of base classes.
    """
    register = get_base_class_register(mixin=mixin, include_baseobjs=include_baseobjs)
    return [x[1] for x in register]


def get_base_class_lookup(mixin=False, include_baseobjs=True):
    """Return lookup dictionary of scitype shorthands to base classes in skpro.

    Parameters
    ----------
    mixin : bool, default=False
        If True, include only mixin scitypes.
        If False, include only non-mixin scitypes.
    include_baseobjs : bool, default=True
        If True, include "object" and "estimator" base scitypes.
        If False, exclude them.

    Returns
    -------
    base_class_lookup : dict
        Dictionary mapping scitype name strings to base classes.

    Examples
    --------
    >>> from skpro.registry._base_classes import get_base_class_lookup
    >>> lookup = get_base_class_lookup()  # doctest: +SKIP
    >>> lookup["regressor_proba"]  # doctest: +SKIP
    <class 'skpro.regression.base._base.BaseProbaRegressor'>
    """
    register = get_base_class_register(mixin=mixin, include_baseobjs=include_baseobjs)
    base_class_lookup = {x[0]: x[1] for x in register}
    return base_class_lookup
