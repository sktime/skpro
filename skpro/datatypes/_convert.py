# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Machine type converters for scitypes.

Exports
-------
convert_to(obj, to_type: str, as_scitype: str, store=None)
    converts object "obj" to type "to_type", considered as "as_scitype"

convert(obj, from_type: str, to_type: str, as_scitype: str, store=None)
    same as convert_to, without automatic identification of "from_type"

mtype(obj, as_scitype: str)
    returns "from_type" of obj, considered as "as_scitype"
---

Function signature of convert

Parameters
----------
obj : object to convert - any type, should comply with mtype spec for as_scitype
from_type : str - the type to convert "obj" to, a valid mtype string
to_type : str - the type to convert "obj" to, a valid mtype string
as_scitype : str - name of scitype the object "obj" is considered as
store : reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_obj : to_type - object obj converted to to_type

---

Function signature of convert_to

Parameters
----------
obj : object to convert - any type, should comply with mtype spec for as_scitype
to_type : str - the type to convert "obj" to, a valid mtype string
as_scitype : str - name of scitype the object "obj" is considered as
store : reference of storage for lossy conversions, default=None (no store)

Returns
-------
converted_obj : to_type - object obj converted to to_type

---

Function signature of mtype

Parameters
----------
obj : object to convert - any type, should comply with mtype spec for as_scitype
as_scitype : str - name of scitype the object "obj" is considered as

Returns
-------
str - the type to convert "obj" to, a valid mtype string
    or None, if obj is None
"""

__author__ = ["fkiraly"]

__all__ = [
    "convert",
    "convert_to",
]

from copy import deepcopy
from functools import lru_cache

import numpy as np
import pandas as pd

from skpro.datatypes._base import BaseConverter
from skpro.datatypes._check import mtype as infer_mtype
from skpro.datatypes._proba import convert_dict_Proba
from skpro.datatypes._registry import mtype_to_scitype
from skpro.datatypes._table import convert_dict_Table


def get_convert_dict(soft_deps="present"):
    """Retrieve convert_dict, caches the first time it is requested.

    This is to avoid repeated, time consuming crawling in generate_check_dict,
    which would otherwise be called every time check_dict is requested.

    Parameters
    ----------
    soft_deps : str, optional - one of "present", "all"
        "present" - only conversions with soft dependencies present are included
        "all" - all conversions are included
    """
    if soft_deps not in ["present", "all"]:
        raise ValueError(
            "Error in get_check_dict, soft_deps argument must be 'present' or 'all', "
            f"found {soft_deps}"
        )
    convert_dict = generate_convert_dict(soft_deps=soft_deps)
    return convert_dict.copy()


@lru_cache(maxsize=1)
def generate_convert_dict(soft_deps="present"):
    """Generate convert_dict using lookup."""
    from skbase.utils.dependencies import _check_estimator_deps

    from skpro.utils.retrieval import _all_classes

    classes = _all_classes("skpro.datatypes")
    classes = [x[1] for x in classes]
    classes = [x for x in classes if issubclass(x, BaseConverter)]
    classes = [x for x in classes if not x.__name__.startswith("Base")]

    # subset only to data types with soft dependencies present
    if soft_deps == "present":
        classes = [x for x in classes if _check_estimator_deps(x, severity="none")]

    convert_dict = dict()
    for cls in classes:
        if not cls.get_class_tag("multiple_conversions", False):
            k = cls()
            key = k._get_key()
            convert_dict[key] = k
        else:
            for cls_to_cls in cls.get_conversions():
                k = cls(*cls_to_cls)

                # check dependencies for both classes
                # only add conversions if dependencies are satisfied for to and from
                cls_from, cls_to = k._get_cls_from_to()

                # do not add conversion if dependencies are not satisfied
                if cls_from is None or cls_to is None:
                    continue
                filter_sd = soft_deps in ["present"]
                if filter_sd and not _check_estimator_deps(cls_from, severity="none"):
                    continue
                if filter_sd and not _check_estimator_deps(cls_to, severity="none"):
                    continue

                key = k._get_key()
                convert_dict[key] = k

    # temporary while refactoring
    convert_dict.update(convert_dict_Proba)
    convert_dict.update(convert_dict_Table)

    return convert_dict


def convert(
    obj,
    from_type: str,
    to_type: str,
    as_scitype: str = None,
    store=None,
    store_behaviour: str = None,
    return_to_mtype: bool = False,
):
    """Convert objects between different machine representations, subject to scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    from_type : str - the type to convert "obj" to, a valid mtype string
        valid mtype strings, with explanation, are in datatypes.MTYPE_REGISTER
    to_type : str - the mtype to convert "obj" to, a valid mtype string
        or list of str, this specifies admissible types for conversion to;
        if list, will convert to first mtype of the same scitype as from_mtype
        valid mtype strings, with explanation, are in datatypes.MTYPE_REGISTER
    as_scitype : str, optional - name of scitype the object "obj" is considered as
        default = inferred from from_type
        valid scitype strings, with explanation, are in datatypes.SCITYPE_REGISTER
    store : optional, reference of storage for lossy conversions, default=None (no ref)
        is updated by side effect if not None and store_behaviour="reset" or "update"
    store_behaviour : str, optional, one of None (default), "reset", "freeze", "update"
        "reset" - store is emptied and then updated from conversion
        "freeze" - store is read-only, may be read/used by conversion but not changed
        "update" - store is updated from conversion and retains previous contents
        None - automatic: "update" if store is empty and not None; "freeze", otherwise
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to

    Returns
    -------
    converted_obj : to_type - object obj converted to to_type
                    if obj was None, returns None
    to_type : str, only returned if ``return_to_mtype=True``
        mtype of ``converted_obj`` - useful of ``to_type`` was a list

    Raises
    ------
    KeyError if conversion is not implemented
    TypeError or ValueError if inputs do not match specification
    """
    if obj is None:
        return None

    # if to_type is a list, we do the following:
    # if on the list, then don't do a conversion (convert to from_type)
    # if not on the list, we find and convert to first mtype that has same scitype
    to_type = _get_first_mtype_of_same_scitype(
        from_mtype=from_type, to_mtypes=to_type, varname="to_type"
    )

    # input type checks
    if not (
        isinstance(to_type, list) and all(isinstance(item, str) for item in to_type)
    ) and not isinstance(to_type, str):
        raise TypeError("to_type must be a str or list of str")
    if not isinstance(from_type, str):
        raise TypeError("from_type must be a str")
    if as_scitype is None:
        as_scitype = mtype_to_scitype(to_type)
    elif not isinstance(as_scitype, str):
        raise TypeError("as_scitype must be str or None")
    if store is not None and not isinstance(store, dict):
        raise TypeError("store must be a dict or None")
    if store_behaviour not in [None, "reset", "freeze", "update"]:
        raise ValueError(
            'store_behaviour must be one of "reset", "freeze", "update", or None'
        )
    if store_behaviour is None and store == {}:
        store_behaviour = "update"
    if store_behaviour is None and store != {}:
        store_behaviour = "freeze"

    key = (from_type, to_type, as_scitype)

    convert_dict = get_convert_dict()
    if key not in convert_dict.keys():
        raise NotImplementedError(
            "no conversion defined from type " + str(from_type) + " to " + str(to_type)
        )

    if store_behaviour == "freeze":
        store = deepcopy(store)
    elif store_behaviour == "reset":
        # note: this is a side effect on store
        store.clear()
    elif store_behaviour == "update":
        # store is passed to convert_obj by reference, unchanged
        # this "elif" is here for clarity, to cover all three values
        pass
    else:
        raise RuntimeError(
            "bug: unreachable condition error, store_behaviour has unexpected value"
        )

    converted_obj = convert_dict[key](obj, store=store)

    if return_to_mtype:
        return converted_obj, to_type
    else:
        return converted_obj


# conversion based on queryable type to specified target
def convert_to(
    obj,
    to_type: str,
    as_scitype: str = None,
    store=None,
    store_behaviour: str = None,
    return_to_mtype: bool = False,
):
    """Convert object to a different machine representation, subject to scitype.

    Parameters
    ----------
    obj : object to convert - any type, should comply with mtype spec for as_scitype
    to_type : str - the mtype to convert "obj" to, a valid mtype string
            or list of str, this specifies admissible types for conversion to;
            if list, will convert to first mtype of the same scitype as obj
        valid mtype strings, with explanation, are in datatypes.MTYPE_REGISTER
    as_scitype : str, optional - name of scitype the object "obj" is considered as
        pre-specifying the scitype reduces the number of checks done in type inference
        valid scitype strings, with explanation, are in datatypes.SCITYPE_REGISTER
        default = inferred from mtype of obj, which is in turn inferred internally
    store : reference of storage for lossy conversions, default=None (no store)
        is updated by side effect if not None and store_behaviour="reset" or "update"
    store_behaviour : str, optional, one of None (default), "reset", "freeze", "update"
        "reset" - store is emptied and then updated from conversion
        "freeze" - store is read-only, may be read/used by conversion but not changed
        "update" - store is updated from conversion and retains previous contents
        None - automatic: "update" if store is empty and not None; "freeze", otherwise
    return_to_mtype: bool, optional (default=False)
        if True, also returns the str of the mtype converted to
    Returns
    -------
    converted_obj : to_type - object obj, or obj converted to target mtype as follows:
        case 1: mtype of obj is equal to to_type, or a list element of to_type
            no conversion happens, converted_obj = obj
        case 2: to_type is a str, and not equal to mtype of obj
            converted_obj is obj converted to to_type
        case 3: to_type is list of str, and mtype of obj is not in that list
            converted_obj is converted to the first mtype in to_type
                that is of same scitype as obj
        case 4: if obj was None, converted_obj is also None
    to_type : str, only returned if ``return_to_mtype=True``
        mtype of ``converted_obj`` - useful of ``to_type`` was a list

    Raises
    ------
    TypeError if machine type of input "obj" is not recognized
    TypeError if to_type contains no mtype compatible with mtype of obj
    KeyError if conversion that would be conducted is not implemented
    TypeError or ValueError if inputs do not match specification
    """
    if obj is None:
        return None

    # input checks on to_type, as_scitype; coerce to_type, as_scitype to lists
    to_type = _check_str_or_list_of_str(to_type, obj_name="to_type")

    # sub-set a preliminary set of as_scitype from to_type, as_scitype
    if as_scitype is not None:
        # if not None, subset to types compatible between to_type and as_scitype
        as_scitype = _check_str_or_list_of_str(as_scitype, obj_name="as_scitype")
        potential_scitypes = mtype_to_scitype(to_type)
        as_scitype = list(set(potential_scitypes).intersection(as_scitype))
    else:
        # if None, infer from to_type
        as_scitype = mtype_to_scitype(to_type)

    # now further narrow down as_scitype by inference from the obj
    from_type = infer_mtype(obj=obj, as_scitype=as_scitype)
    as_scitype = mtype_to_scitype(from_type)

    converted_obj = convert(
        obj=obj,
        from_type=from_type,
        to_type=to_type,
        as_scitype=as_scitype,
        store=store,
        store_behaviour=store_behaviour,
        return_to_mtype=return_to_mtype,
    )

    return converted_obj


def _get_first_mtype_of_same_scitype(from_mtype, to_mtypes, varname="to_mtypes"):
    """Return first mtype in list mtypes that has same scitype as from_mtype.

    Parameters
    ----------
    from_mtype : str - mtype of object to convert from
    to_mtypes : list of str - mtypes to convert to
    varname : str - name of variable to_mtypes, for error message

    Returns
    -------
    to_type : str - first mtype in to_mtypes that has same scitype as from_mtype
    """
    to_mtypes = _check_str_or_list_of_str(to_mtypes, obj_name=varname)
    if not isinstance(to_mtypes, list):
        raise TypeError(f"{varname} must be a str or a list of str")
    # no conversion of from_type is in the list
    if from_mtype in to_mtypes:
        return from_mtype
    # otherwise convert to first element of same scitype
    scitype = mtype_to_scitype(from_mtype)
    same_scitype_mtypes = [
        mtype for mtype in to_mtypes if mtype_to_scitype(mtype) == scitype
    ]
    if len(same_scitype_mtypes) == 0:
        raise TypeError(
            f"{varname} contains no mtype compatible with the scitype of obj, "
            f"which is {scitype}. Value of {varname} is: {to_mtypes}"
        )
    to_type = same_scitype_mtypes[0]
    return to_type


def _conversions_defined(scitype: str, soft_deps: str = "present"):
    """Return an indicator matrix which conversions are defined for scitype.

    Parameters
    ----------
    scitype: str - name of scitype for which conversions are queried
        valid scitype strings, with explanation, are in datatypes.SCITYPE_REGISTER
    soft_deps : str, optional - one of "present", "all"
        "present" - only conversions with soft dependencies present are included
        "all" - all conversions are included

    Returns
    -------
    conv_df: pd.DataFrame, columns and index is list of mtypes for scitype
            entry of row i, col j is 1 if conversion from i to j is defined,
                                     0 if conversion from i to j is not defined
    """
    convert_dict = get_convert_dict(soft_deps=soft_deps)
    pairs = [(x[0], x[1]) for x in list(convert_dict.keys()) if x[2] == scitype]
    cols0 = {x[0] for x in list(convert_dict.keys()) if x[2] == scitype}
    cols1 = {x[1] for x in list(convert_dict.keys()) if x[2] == scitype}
    cols = sorted(list(cols0.union(cols1)))

    mat = np.zeros((len(cols), len(cols)), dtype=int)
    nkeys = len(cols)
    for i in range(nkeys):
        for j in range(nkeys):
            if (cols[i], cols[j]) in pairs:
                mat[i, j] = 1

    conv_df = pd.DataFrame(mat, index=cols, columns=cols)

    return conv_df


def _check_str_or_list_of_str(obj, obj_name="obj"):
    """Check whether obj is str or list of str; coerces to list of str."""
    if isinstance(obj, list):
        if not np.all([isinstance(x, str) for x in obj]):
            raise TypeError(f"{obj} must be a str or list of str")
        else:
            return obj
    elif not isinstance(obj, str):
        raise TypeError(f"{obj} must be a str or list of str")
    else:
        return [obj]
