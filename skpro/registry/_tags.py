"""Register of estimator and object tags.

Note for extenders: new tags should be entered as classes inheriting from _BaseTag.
No other place is necessary to add new tags.

This module exports the following:

---
OBJECT_TAG_REGISTER - list of tuples

each tuple corresponds to a tag, elements as follows:
    0 : string - name of the tag as used in the _tags dictionary
    1 : string - name of the scitype this tag applies to
                 must be in _base_classes.BASE_CLASS_SCITYPE_LIST
    2 : string - expected type of the tag value
        should be one of:
            "bool" - valid values are True/False
            "int" - valid values are all integers
            "str" - valid values are all strings
            "list" - valid values are all lists of arbitrary elements
            ("str", list_of_string) - any string in list_of_string is valid
            ("list", list_of_string) - any individual string and sub-list is valid
            ("list", "str") - any individual string or list of strings is valid
        validity can be checked by check_tag_is_valid (see below)
    3 : string - plain English description of the tag

---

OBJECT_TAG_TABLE - pd.DataFrame
    OBJECT_TAG_REGISTER in table form, as pd.DataFrame
        rows of OBJECT_TABLE correspond to elements in OBJECT_TAG_REGISTER

OBJECT_TAG_LIST - list of string
    elements are 0-th entries of OBJECT_TAG_REGISTER, in same order

---

check_tag_is_valid(tag_name, tag_value) - checks whether tag_value is valid for tag_name
"""
import inspect
import sys

import pandas as pd

from skpro.base import BaseObject


class _BaseTag(BaseObject):
    """Base class for all tags.

    All tags in skpro should inherit from this class, and set the ``_tags``
    class variable to a dictionary with the following keys:

    - ``"tag_name"``: string, the name of the tag as used in ``_tags`` dicts
    - ``"parent_type"``: string or list of str, scitype(s) this tag applies to
    - ``"tag_type"``: string or tuple, the expected type of the tag value
    - ``"short_descr"``: string, short English description of the tag
    - ``"user_facing"``: bool, whether the tag is user-facing
    """

    _tags = {
        "object_type": "tag",
        "tag_name": "fill_this_in",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "describe the tag here",
        "user_facing": True,
    }


# --------------------------
# all objects and estimators
# --------------------------
class reserved_params(_BaseTag):
    """List of reserved parameter names for the object.

    - String name: ``"reserved_params"``
    - Private tag, developer and framework facing
    - Values: list of strings
    """

    _tags = {
        "tag_name": "reserved_params",
        "parent_type": "object",
        "tag_type": "list",
        "short_descr": "list of reserved parameter names",
        "user_facing": False,
    }


class object_type(_BaseTag):
    """Scientific type of the object.

    - String name: ``"object_type"``
    - Public metadata tag
    - Values: string, e.g., ``"regressor"``, ``"transformer"``
    """

    _tags = {
        "tag_name": "object_type",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "type of object, e.g., 'regressor', 'transformer'",
        "user_facing": True,
    }


class estimator_type(_BaseTag):
    """Type of estimator.

    - String name: ``"estimator_type"``
    - Public metadata tag
    - Values: string, e.g., ``"regressor"``, ``"transformer"``
    """

    _tags = {
        "tag_name": "estimator_type",
        "parent_type": "estimator",
        "tag_type": "str",
        "short_descr": "type of estimator, e.g., 'regressor', 'transformer'",
        "user_facing": True,
    }


# ---------------------
# packaging information
# ---------------------
class maintainers(_BaseTag):
    """Current maintainers of the object, GitHub IDs.

    Part of packaging metadata for the object.

    - String name: ``"maintainers"``
    - Public metadata tag
    - Values: string or list of strings, each a GitHub handle
    - Example: ``["fkiraly", "yarnabrina"]``
    """

    _tags = {
        "tag_name": "maintainers",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": (
            "list of current maintainers of the object, "
            "each maintainer a GitHub handle"
        ),
        "user_facing": True,
    }


class authors(_BaseTag):
    """Authors of the object, GitHub IDs.

    Part of packaging metadata for the object.

    - String name: ``"authors"``
    - Public metadata tag
    - Values: string or list of strings, each a GitHub handle
    - Example: ``["fkiraly"]``
    """

    _tags = {
        "tag_name": "authors",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of authors of the object, each author a GitHub handle",
        "user_facing": True,
    }


class python_version(_BaseTag):
    """Python version specifier for the object (PEP 440).

    Part of packaging metadata for the object.

    - String name: ``"python_version"``
    - Private tag, developer and framework facing
    - Values: PEP 440 compliant version specifier, or None
    - Example: ``">=3.10"``
    """

    _tags = {
        "tag_name": "python_version",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": (
            "python version specifier (PEP 440) for estimator, "
            "or None = all versions ok"
        ),
        "user_facing": False,
    }


class python_dependencies(_BaseTag):
    """Python dependencies of the object.

    Part of packaging metadata for the object.

    - String name: ``"python_dependencies"``
    - Private tag, developer and framework facing
    - Values: string or list of strings (PEP 440 specifiers)
    - Example: ``["numpy>=1.21", "scipy"]``
    """

    _tags = {
        "tag_name": "python_dependencies",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "python dependencies of estimator as str or list of str",
        "user_facing": False,
    }


class python_dependencies_alias(_BaseTag):
    """Alias mapping for python dependencies when import name differs from package.

    - String name: ``"python_dependencies_alias"``
    - Private tag, developer and framework facing
    - Values: dict, key-value pairs of package name to import name
    """

    _tags = {
        "tag_name": "python_dependencies_alias",
        "parent_type": "object",
        "tag_type": "dict",
        "short_descr": (
            "should be provided if import name differs from package name, "
            "key-value pairs are package name, import name"
        ),
        "user_facing": False,
    }


class license_type(_BaseTag):
    """License type for interfaced packages.

    - String name: ``"license_type"``
    - Public metadata tag
    - Values: string, one of ``"copyleft"``, ``"permissive"``, ``"copyright"``

    .. warning::
        May be incorrect. NO LIABILITY assumed for this field.
    """

    _tags = {
        "tag_name": "license_type",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": (
            "license type for interfaced packages: "
            "'copyleft', 'permissive', 'copyright'. "
            "may be incorrect, NO LIABILITY assumed for this field"
        ),
        "user_facing": True,
    }


# -----------------
# CI and test flags
# -----------------
class tests__libs(_BaseTag):
    """Library dependencies required for tests.

    - String name: ``"tests:libs"``
    - Private tag, developer and framework facing
    - Values: list of strings
    """

    _tags = {
        "tag_name": "tests:libs",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of library dependencies required for tests",
        "user_facing": False,
    }


class tests__vm(_BaseTag):
    """Whether tests require their own VM to run.

    - String name: ``"tests:vm"``
    - Private tag, developer and framework facing
    - Values: boolean, ``True`` / ``False``
    """

    _tags = {
        "tag_name": "tests:vm",
        "parent_type": "object",
        "tag_type": "bool",
        "short_descr": "whether tests require their own VM to run",
        "user_facing": False,
    }


class tests__skip_by_name(_BaseTag):
    """List of test names to skip when running estimator checks on CI.

    - String name: ``"tests:skip_by_name"``
    - Private tag, developer and framework facing
    - Values: list of strings
    """

    _tags = {
        "tag_name": "tests:skip_by_name",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": (
            "list of test names to skip when running estimator checks on CI"
        ),
        "user_facing": False,
    }


class tests__python_dependencies(_BaseTag):
    """Additional python dependencies needed in tests.

    - String name: ``"tests:python_dependencies"``
    - Private tag, developer and framework facing
    - Values: string or list of strings (PEP 440 specifiers)
    """

    _tags = {
        "tag_name": "tests:python_dependencies",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": (
            "additional python dependencies needed in tests, "
            "str or list of str (PEP 440)"
        ),
        "user_facing": False,
    }


# ------------------
# BaseProbaRegressor
# ------------------
class capability__survival(_BaseTag):
    """Whether estimator can use censoring information, for survival analysis.

    - String name: ``"capability:survival"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Default: ``False``
    """

    _tags = {
        "tag_name": "capability:survival",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": (
            "whether estimator can use censoring information, "
            "for survival analysis"
        ),
        "user_facing": True,
    }


class capability__multioutput(_BaseTag):
    """Whether estimator supports multioutput regression.

    - String name: ``"capability:multioutput"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Default: ``False``
    """

    _tags = {
        "tag_name": "capability:multioutput",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports multioutput regression",
        "user_facing": True,
    }


class capability__missing(_BaseTag):
    """Whether estimator supports missing values.

    - String name: ``"capability:missing"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Default: ``False``
    """

    _tags = {
        "tag_name": "capability:missing",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports missing values",
        "user_facing": True,
    }


class capability__update(_BaseTag):
    """Whether estimator supports online updates via update.

    - String name: ``"capability:update"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``
    - Default: ``False``
    """

    _tags = {
        "tag_name": "capability:update",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports online updates via update",
        "user_facing": True,
    }


class X_inner_mtype(_BaseTag):
    """Internal machine type(s) for X in _fit/_predict.

    - String name: ``"X_inner_mtype"``
    - Private tag, developer and framework facing
    - Values: string or list of strings
    """

    _tags = {
        "tag_name": "X_inner_mtype",
        "parent_type": "regressor_proba",
        "tag_type": ("list", "str"),
        "short_descr": (
            "which machine type(s) is the internal _fit/_predict "
            "able to deal with?"
        ),
        "user_facing": False,
    }


class y_inner_mtype(_BaseTag):
    """Internal machine type(s) for y in _fit/_predict.

    - String name: ``"y_inner_mtype"``
    - Private tag, developer and framework facing
    - Values: string or list of strings
    """

    _tags = {
        "tag_name": "y_inner_mtype",
        "parent_type": "regressor_proba",
        "tag_type": ("list", "str"),
        "short_descr": (
            "which machine type(s) is the internal _fit/_predict "
            "able to deal with?"
        ),
        "user_facing": False,
    }


class C_inner_mtype(_BaseTag):
    """Internal machine type(s) for C in _fit/_predict.

    - String name: ``"C_inner_mtype"``
    - Private tag, developer and framework facing
    - Values: string or list of strings
    """

    _tags = {
        "tag_name": "C_inner_mtype",
        "parent_type": "regressor_proba",
        "tag_type": ("list", "str"),
        "short_descr": (
            "which machine type(s) is the internal _fit/_predict "
            "able to deal with?"
        ),
        "user_facing": False,
    }


# ----------------
# BaseDistribution
# ----------------
class capabilities__approx(_BaseTag):
    """Methods of distribution that are approximate.

    - String name: ``"capabilities:approx"``
    - Private tag, developer and framework facing
    - Values: list of strings
    """

    _tags = {
        "tag_name": "capabilities:approx",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "methods of distr that are approximate",
        "user_facing": False,
    }


class capabilities__exact(_BaseTag):
    """Methods of distribution that are numerically exact.

    - String name: ``"capabilities:exact"``
    - Private tag, developer and framework facing
    - Values: list of strings
    """

    _tags = {
        "tag_name": "capabilities:exact",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "methods of distr that are numerically exact",
        "user_facing": False,
    }


class distr__measuretype(_BaseTag):
    """Measure type of distribution.

    - String name: ``"distr:measuretype"``
    - Public metadata tag
    - Values: one of ``"continuous"``, ``"discrete"``, ``"mixed"``
    """

    _tags = {
        "tag_name": "distr:measuretype",
        "parent_type": "distribution",
        "tag_type": ("str", ["continuous", "discrete", "mixed"]),
        "short_descr": "measure type of distr",
        "user_facing": True,
    }


class distr__paramtype(_BaseTag):
    """Parametrization type of distribution.

    - String name: ``"distr:paramtype"``
    - Public metadata tag
    - Values: one of ``"general"``, ``"parametric"``, ``"nonparametric"``,
      ``"composite"``
    """

    _tags = {
        "tag_name": "distr:paramtype",
        "parent_type": "distribution",
        "tag_type": (
            "str",
            ["general", "parametric", "nonparametric", "composite"],
        ),
        "short_descr": "parametrization type of distribution",
        "user_facing": True,
    }


class approx_mean_spl(_BaseTag):
    """Sample size used in Monte Carlo estimates of mean.

    - String name: ``"approx_mean_spl"``
    - Private tag, developer and framework facing
    - Values: integer
    """

    _tags = {
        "tag_name": "approx_mean_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of mean",
        "user_facing": False,
    }


class approx_var_spl(_BaseTag):
    """Sample size used in Monte Carlo estimates of var.

    - String name: ``"approx_var_spl"``
    - Private tag, developer and framework facing
    - Values: integer
    """

    _tags = {
        "tag_name": "approx_var_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of var",
        "user_facing": False,
    }


class approx_energy_spl(_BaseTag):
    """Sample size used in Monte Carlo estimates of energy.

    - String name: ``"approx_energy_spl"``
    - Private tag, developer and framework facing
    - Values: integer
    """

    _tags = {
        "tag_name": "approx_energy_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of energy",
        "user_facing": False,
    }


class approx_spl(_BaseTag):
    """Sample size used in other Monte Carlo estimates.

    - String name: ``"approx_spl"``
    - Private tag, developer and framework facing
    - Values: integer
    """

    _tags = {
        "tag_name": "approx_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in other MC estimates",
        "user_facing": False,
    }


class bisect_iter(_BaseTag):
    """Max iterations for bisection method in ppf.

    - String name: ``"bisect_iter"``
    - Private tag, developer and framework facing
    - Values: integer
    """

    _tags = {
        "tag_name": "bisect_iter",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "max iters for bisection method in ppf",
        "user_facing": False,
    }


class broadcast_params(_BaseTag):
    """Distribution parameters to broadcast.

    Complement of broadcast parameters is not broadcast.

    - String name: ``"broadcast_params"``
    - Private tag, developer and framework facing
    - Values: list of strings
    """

    _tags = {
        "tag_name": "broadcast_params",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": (
            "distribution parameters to broadcast, "
            "complement is not broadcast"
        ),
        "user_facing": False,
    }


class broadcast_init(_BaseTag):
    """Whether to initialize broadcast parameters in __init__.

    - String name: ``"broadcast_init"``
    - Private tag, developer and framework facing
    - Values: ``"on"`` or ``"off"``
    """

    _tags = {
        "tag_name": "broadcast_init",
        "parent_type": "distribution",
        "tag_type": ("str", ["on", "off"]),
        "short_descr": (
            "whether to initialize broadcast parameters in __init__, "
            "'on' or 'off'"
        ),
        "user_facing": False,
    }


class broadcast_inner(_BaseTag):
    """Whether inner logic is vectorized or scalar.

    - String name: ``"broadcast_inner"``
    - Private tag, developer and framework facing
    - Values: ``"array"`` or ``"scalar"``
    """

    _tags = {
        "tag_name": "broadcast_inner",
        "parent_type": "distribution",
        "tag_type": ("str", ["array", "scalar"]),
        "short_descr": (
            "if inner logic is vectorized ('array') or scalar ('scalar')"
        ),
        "user_facing": False,
    }


# ---------------
# BaseProbaMetric
# ---------------
class scitype__y_pred(_BaseTag):
    """Expected input type for y_pred in performance metric.

    - String name: ``"scitype:y_pred"``
    - Public metadata tag
    - Values: string
    """

    _tags = {
        "tag_name": "scitype:y_pred",
        "parent_type": "metric",
        "tag_type": "str",
        "short_descr": "expected input type for y_pred in performance metric",
        "user_facing": True,
    }


class lower_is_better(_BaseTag):
    """Whether lower or higher metric values are better.

    - String name: ``"lower_is_better"``
    - Public metadata tag
    - Values: boolean, ``True`` (lower is better) / ``False`` (higher is better)
    """

    _tags = {
        "tag_name": "lower_is_better",
        "parent_type": "metric",
        "tag_type": "bool",
        "short_descr": "whether lower (True) or higher (False) is better",
        "user_facing": True,
    }


class capability__survival_metric(_BaseTag):
    """Whether metric uses censoring information, for survival analysis.

    - String name: ``"capability:survival"``
    - Public capability tag
    - Values: boolean, ``True`` / ``False``

    Note: this tag shares the name ``"capability:survival"`` with the
    regressor tag, but applies to the ``"metric"`` parent type.
    """

    _tags = {
        "tag_name": "capability:survival",
        "parent_type": "metric",
        "tag_type": "bool",
        "short_descr": (
            "whether metric uses censoring information, "
            "for survival analysis"
        ),
        "user_facing": True,
    }


# ----------------------------
# BaseMetaObject reserved tags
# ----------------------------
class named_object_parameters(_BaseTag):
    """Name of component list attribute for meta-objects.

    - String name: ``"named_object_parameters"``
    - Private tag, developer and framework facing
    - Values: string
    """

    _tags = {
        "tag_name": "named_object_parameters",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "name of component list attribute for meta-objects",
        "user_facing": False,
    }


class fitted_named_object_parameters(_BaseTag):
    """Name of fitted component list attribute for meta-objects.

    - String name: ``"fitted_named_object_parameters"``
    - Private tag, developer and framework facing
    - Values: string
    """

    _tags = {
        "tag_name": "fitted_named_object_parameters",
        "parent_type": "estimator",
        "tag_type": "str",
        "short_descr": (
            "name of fitted component list attribute for meta-objects"
        ),
        "user_facing": False,
    }


# -------------------------------------------------------
# construct the tag register from all classes in this module
# -------------------------------------------------------
OBJECT_TAG_REGISTER = []

tag_clses = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for _, cl in tag_clses:
    # skip the base class and non-tag classes
    if cl.__name__ == "_BaseTag" or not issubclass(cl, _BaseTag):
        continue

    cl_tags = cl.get_class_tags()

    tag_name = cl_tags["tag_name"]
    parent_type = cl_tags["parent_type"]
    tag_type = cl_tags["tag_type"]
    short_descr = cl_tags["short_descr"]

    OBJECT_TAG_REGISTER.append((tag_name, parent_type, tag_type, short_descr))

OBJECT_TAG_TABLE = pd.DataFrame(OBJECT_TAG_REGISTER)
OBJECT_TAG_LIST = OBJECT_TAG_TABLE[0].tolist()


def check_tag_is_valid(tag_name, tag_value):
    """Check validity of a tag value.

    Parameters
    ----------
    tag_name : string, name of the tag
    tag_value : object, value of the tag

    Raises
    ------
    KeyError - if tag_name is not a valid tag in OBJECT_TAG_LIST
    ValueError - if the tag_valid is not a valid for the tag with name tag_name
    """
    if tag_name not in OBJECT_TAG_LIST:
        raise KeyError(tag_name + " is not a valid tag")

    tag_type = OBJECT_TAG_TABLE[2][OBJECT_TAG_TABLE[0] == "tag_name"]

    if tag_type == "bool" and not isinstance(tag_value, bool):
        raise ValueError(tag_name + " must be True/False, found " + tag_value)

    if tag_type == "int" and not isinstance(tag_value, int):
        raise ValueError(tag_name + " must be integer, found " + tag_value)

    if tag_type == "str" and not isinstance(tag_value, str):
        raise ValueError(tag_name + " must be string, found " + tag_value)

    if tag_type == "list" and not isinstance(tag_value, list):
        raise ValueError(tag_name + " must be list, found " + tag_value)

    if tag_type[0] == "str" and tag_value not in tag_type[1]:
        raise ValueError(
            tag_name + " must be one of " + tag_type[1] + " found " + tag_value
        )

    if tag_type[0] == "list" and not set(tag_value).issubset(tag_type[1]):
        raise ValueError(
            tag_name + " must be subest of " + tag_type[1] + " found " + tag_value
        )

    if tag_type[0] == "list" and tag_type[1] == "str":
        msg = f"{tag_name} must be str or list of str, found {tag_value}"
        if not isinstance(tag_value, (str, list)):
            raise ValueError(msg)
        if isinstance(tag_value, list):
            if not all(isinstance(x, str) for x in tag_value):
                raise ValueError(msg)
