"""Registry lookup methods.

This module exports the following methods for registry lookup:

all_objects(object_types, filter_tags)
    lookup and filtering of objects
"""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
# based on the sktime module of same name

__author__ = ["fkiraly"]
# all_objects is based on the sklearn utility all_estimators


from copy import deepcopy
from operator import itemgetter
from pathlib import Path

import pandas as pd
from skbase.lookup import all_objects as _all_objects

from skpro.base import BaseEstimator, BaseObject
from skpro.registry._tags import OBJECT_TAG_REGISTER

VALID_OBJECT_TYPE_STRINGS = {x[1] for x in OBJECT_TAG_REGISTER}


def all_objects(
    object_types=None,
    filter_tags=None,
    exclude_objects=None,
    return_names=True,
    as_dataframe=False,
    return_tags=None,
    suppress_import_stdout=True,
):
    """Get a list of all objects from skpro.

    This function crawls the module and gets all classes that inherit
    from skpro's and sklearn's base classes.

    Not included are: the base classes themselves, classes defined in test
    modules.

    Parameters
    ----------
    object_types: str, list of str, optional (default=None)
        Which kind of objects should be returned.
        if None, no filter is applied and all objects are returned.
        if str or list of str, strings define scitypes specified in search
        only objects that are of (at least) one of the scitypes are returned
        possible str values are entries of registry.BASE_CLASS_REGISTER (first col)
        for instance 'regrssor_proba', 'distribution, 'metric'

    return_names: bool, optional (default=True)

        if True, estimator class name is included in the ``all_objects``
        return in the order: name, estimator class, optional tags, either as
        a tuple or as pandas.DataFrame columns

        if False, estimator class name is removed from the ``all_objects`` return.

    filter_tags: dict of (str or list of str), optional (default=None)
        For a list of valid tag strings, use the registry.all_tags utility.

        ``filter_tags`` subsets the returned estimators as follows:

        * each key/value pair is statement in "and"/conjunction
        * key is tag name to sub-set on
        * value str or list of string are tag values
        * condition is "key must be equal to value, or in set(value)"

    exclude_estimators: str, list of str, optional (default=None)
        Names of estimators to exclude.

    as_dataframe: bool, optional (default=False)

        True: ``all_objects`` will return a pandas.DataFrame with named
        columns for all of the attributes being returned.

        False: ``all_objects`` will return a list (either a list of
        estimators or a list of tuples, see Returns)

    return_tags: str or list of str, optional (default=None)
        Names of tags to fetch and return each estimator's value of.
        For a list of valid tag strings, use the registry.all_tags utility.
        if str or list of str,
        the tag values named in return_tags will be fetched for each
        estimator and will be appended as either columns or tuple entries.

    suppress_import_stdout : bool, optional. Default=True
        whether to suppress stdout printout upon import.

    Returns
    -------
    all_objects will return one of the following:
        1. list of objects, if return_names=False, and return_tags is None
        2. list of tuples (optional object name, class, ~optional object
          tags), if return_names=True or return_tags is not None.
        3. pandas.DataFrame if as_dataframe = True
        if list of objects:
            entries are objects matching the query,
            in alphabetical order of object name
        if list of tuples:
            list of (optional object name, object, optional object
            tags) matching the query, in alphabetical order of object name,
            where
            ``name`` is the object name as string, and is an
                optional return
            ``object`` is the actual object
            ``tags`` are the object's values for each tag in return_tags
                and is an optional return.
        if dataframe:
            all_objects will return a pandas.DataFrame.
            column names represent the attributes contained in each column.
            "objects" will be the name of the column of objects, "names"
            will be the name of the column of object class names and the string(s)
            passed in return_tags will serve as column names for all columns of
            tags that were optionally requested.

    Examples
    --------
    >>> from skpro.registry import all_objects
    >>> # return a complete list of objects as pd.Dataframe
    >>> all_objects(as_dataframe=True)
    >>> # return all probabilistic regressors by filtering for object type
    >>> all_objects("regressor_proba", as_dataframe=True)
    >>> # return all regressors which handle missing data in the input by tag filtering
    >>> all_objects(
    ...     "regressor_proba",
    ...     filter_tags={"capability:missing": True},
    ...     as_dataframe=True
    ... )

    References
    ----------
    Adapted version of sktime's ``all_estimators``,
    which is an evolution of scikit-learn's ``all_estimators``
    """
    MODULES_TO_IGNORE = (
        "tests",
        "setup",
        "contrib",
        "utils",
        "all",
    )

    result = []
    ROOT = str(Path(__file__).parent.parent)  # skpro package root directory

    if isinstance(filter_tags, str):
        filter_tags = {filter_tags: True}
    filter_tags = filter_tags.copy() if filter_tags else None

    if object_types:
        if filter_tags and "object_type" not in filter_tags.keys():
            object_tag_filter = {"object_type": object_types}
        elif filter_tags:
            filter_tags_filter = filter_tags.get("object_type", [])
            if isinstance(object_types, str):
                object_types = [object_types]
            object_tag_update = {"object_type": object_types + filter_tags_filter}
            filter_tags.update(object_tag_update)
        else:
            object_tag_filter = {"object_type": object_types}
        if filter_tags:
            filter_tags.update(object_tag_filter)
        else:
            filter_tags = object_tag_filter

    result = _all_objects(
        object_types=[BaseObject, BaseEstimator],
        filter_tags=filter_tags,
        exclude_objects=exclude_objects,
        return_names=return_names,
        as_dataframe=as_dataframe,
        return_tags=return_tags,
        suppress_import_stdout=suppress_import_stdout,
        package_name="skpro",
        path=ROOT,
        modules_to_ignore=MODULES_TO_IGNORE,
    )

    return result


def _check_list_of_str_or_error(arg_to_check, arg_name):
    """Check that certain arguments are str or list of str.

    Parameters
    ----------
    arg_to_check: argument we are testing the type of
    arg_name: str,
        name of the argument we are testing, will be added to the error if
        ``arg_to_check`` is not a str or a list of str

    Returns
    -------
    arg_to_check: list of str,
        if arg_to_check was originally a str it converts it into a list of str
        so that it can be iterated over.

    Raises
    ------
    TypeError if arg_to_check is not a str or list of str
    """
    # check that return_tags has the right type:
    if isinstance(arg_to_check, str):
        arg_to_check = [arg_to_check]
    if not isinstance(arg_to_check, list) or not all(
        isinstance(value, str) for value in arg_to_check
    ):
        raise TypeError(
            f"Error in all_objects!  Argument {arg_name} must be either\
             a str or list of str"
        )
    return arg_to_check


def _get_return_tags(object, return_tags):
    """Fetch a list of all tags for every_entry of all_objects.

    Parameters
    ----------
    object: BaseObject, an skpro object
    return_tags: list of str,
        names of tags to get values for the object

    Returns
    -------
    tags: a tuple with all the objects values for all tags in return tags.
        a value is None if it is not a valid tag for the object provided.
    """
    tags = tuple(object.get_class_tag(tag) for tag in return_tags)
    return tags


def _check_tag_cond(object, filter_tags=None, as_dataframe=True):
    """Check whether object satisfies filter_tags condition.

    Parameters
    ----------
    object: BaseObject, an skpro object
    filter_tags: dict of (str or list of str), default=None
        subsets the returned objects as follows:
            each key/value pair is statement in "and"/conjunction
                key is tag name to sub-set on
                value str or list of string are tag values
                condition is "key must be equal to value, or in set(value)"
    as_dataframe: bool, default=False
                if False, return is as described below;
                if True, return is converted into a pandas.DataFrame for pretty
                display

    Returns
    -------
    cond_sat: bool, whether object satisfies condition in filter_tags
    """
    if not isinstance(filter_tags, dict):
        raise TypeError("filter_tags must be a dict")

    cond_sat = True

    for key, value in filter_tags.items():
        if not isinstance(value, list):
            value = [value]
        cond_sat = cond_sat and object.get_class_tag(key) in set(value)

    return cond_sat


def all_tags(
    object_types=None,
    as_dataframe=False,
):
    """Get a list of all tags from skpro.

    Retrieves tags directly from `_tags`, offers filtering functionality.

    Parameters
    ----------
    object_types: string, list of string, optional (default=None)
        Which kind of objects should be returned.
        If None, no filter is applied and all objects are returned.
    as_dataframe: bool, optional (default=False)
        if False, return is as described below;
        if True, return is converted into a pandas.DataFrame for pretty display

    Returns
    -------
    tags: list of tuples (a, b, c, d),
        in alphabetical order by a
        a : string - name of the tag as used in the _tags dictionary
        b : string - name of the scitype this tag applies to
                    must be in _base_classes.BASE_CLASS_SCITYPE_LIST
        c : string - expected type of the tag value
            should be one of:
                "bool" - valid values are True/False
                "int" - valid values are all integers
                "str" - valid values are all strings
                ("str", list_of_string) - any string in list_of_string is valid
                ("list", list_of_string) - any individual string and sub-list is valid
        d : string - plain English description of the tag
    """

    def is_tag_for_type(tag, object_types):
        tag_types = tag[1]
        tag_types = _check_list_of_str_or_error(tag_types, "tag_types")

        if isinstance(object_types, str):
            object_types = [object_types]

        tag_types = set(tag_types)
        object_types = set(object_types)
        is_valid_tag_for_type = len(tag_types.intersection(object_types)) > 0

        return is_valid_tag_for_type

    all_tags = OBJECT_TAG_REGISTER

    if object_types:
        # checking, but not using the return since that is classes, not strings
        _check_object_types(object_types)
        all_tags = [tag for tag in all_tags if is_tag_for_type(tag, object_types)]

    all_tags = sorted(all_tags, key=itemgetter(0))

    # convert to pd.DataFrame if as_dataframe=True
    if as_dataframe:
        columns = ["name", "scitype", "type", "description"]
        all_tags = pd.DataFrame(all_tags, columns=columns)

    return all_tags


def _check_object_types(object_types):
    """Return list of classes corresponding to type strings."""
    object_types = deepcopy(object_types)

    if not isinstance(object_types, list):
        object_types = [object_types]  # make iterable

    def _get_err_msg(object_type):
        return (
            f"Parameter `object_type` must be None, a string or a list of "
            f"strings. Valid string values are: "
            f"{tuple(VALID_OBJECT_TYPE_STRINGS)}, but found: "
            f"{repr(object_type)}"
        )

    for object_type in object_types:
        if not isinstance(object_type, (type, str)):
            raise ValueError(
                "Please specify `object_types` as a list of str or " "types."
            )
        if isinstance(object_type, str):
            if object_type not in VALID_OBJECT_TYPE_STRINGS:
                raise ValueError(_get_err_msg(object_type))
    return object_types
