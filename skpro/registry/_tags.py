"""Register of estimator and object tags.

Note for extenders: new tags should be entered by creating a subclass of _BaseTag.
No other place is necessary to add new tags.

This module exports the following:

---

OBJECT_TAG_REGISTER - list of tuples
    each tuple corresponds to a tag, elements as follows:
        0 : string - name of the tag as used in the _tags dictionary
        1 : string - name of the scitype this tag applies to
        2 : string - expected type of the tag value
        3 : string - plain English description of the tag

OBJECT_TAG_TABLE - pd.DataFrame
    OBJECT_TAG_REGISTER in table form, as pd.DataFrame

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
    """Base class for all tags."""

    _tags = {
        "object_type": "tag",
        "tag_name": "fill_this_in",  # name of the tag used in the _tags dictionary
        "parent_type": "object",  # scitype of the parent object, str or list of str
        "tag_type": "str",  # type of the tag value
        "short_descr": "describe the tag here",  # short tag description, max 80 chars
        "user_facing": True,  # whether the tag is user-facing
    }


# --------------------------
# all objects and estimators
# --------------------------


class reserved_params(_BaseTag):
    """List of reserved parameter names.

    - String name: ``"reserved_params"``
    - Values: ``list``
    """

    _tags = {
        "tag_name": "reserved_params",
        "parent_type": "object",
        "tag_type": "list",
        "short_descr": "list of reserved parameter names",
    }


class object_type(_BaseTag):
    """Type of object, e.g., 'regressor', 'transformer'.

    - String name: ``"object_type"``
    - Values: ``str``
    """

    _tags = {
        "tag_name": "object_type",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "type of object, e.g., 'regressor', 'transformer'",
    }


class estimator_type(_BaseTag):
    """Type of estimator, e.g., 'regressor', 'transformer'.

    - String name: ``"estimator_type"``
    - Values: ``str``
    """

    _tags = {
        "tag_name": "estimator_type",
        "parent_type": "estimator",
        "tag_type": "str",
        "short_descr": "type of estimator, e.g., 'regressor', 'transformer'",
    }


# packaging information
# ---------------------


class maintainers(_BaseTag):
    """Current maintainers of the object, each maintainer a GitHub handle.

    - String name: ``"maintainers"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "maintainers",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": (
            "list of current maintainers of the object, "
            "each maintainer a GitHub handle"
        ),
    }


class authors(_BaseTag):
    """Authors of the object, each author a GitHub handle.

    - String name: ``"authors"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "authors",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of authors of the object, each author a GitHub handle",
    }


class python_version(_BaseTag):
    """Python version specifier (PEP 440) for estimator, or None = all versions ok.

    - String name: ``"python_version"``
    - Values: ``str``
    """

    _tags = {
        "tag_name": "python_version",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": (
            "python version specifier (PEP 440) for estimator, "
            "or None = all versions ok"
        ),
    }


class python_dependencies(_BaseTag):
    """Python dependencies of estimator as str or list of str.

    - String name: ``"python_dependencies"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "python_dependencies",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "python dependencies of estimator as str or list of str",
    }


class python_dependencies_alias(_BaseTag):
    """Alias mapping for package names whose import name differs from package name.

    - String name: ``"python_dependencies_alias"``
    - Values: ``dict``
    """

    _tags = {
        "tag_name": "python_dependencies_alias",
        "parent_type": "object",
        "tag_type": "dict",
        "short_descr": (
            "should be provided if import name differs from package name, "
            "key-value pairs are package name, import name"
        ),
    }


class license_type(_BaseTag):
    """License type for interfaced packages.

    - String name: ``"license_type"``
    - Values: ``str``
    """

    _tags = {
        "tag_name": "license_type",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": (
            "license type for interfaced packages: 'copyleft', 'permissive', "
            "'copyright'. may be incorrect, NO LIABILITY assumed for this field"
        ),
    }


# CI and test flags
# -----------------


class tests__libs(_BaseTag):
    """List of library dependencies required for tests.

    - String name: ``"tests:libs"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "tests:libs",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of library dependencies required for tests",
    }


class tests__vm(_BaseTag):
    """Whether tests require their own VM to run.

    - String name: ``"tests:vm"``
    - Values: ``bool``
    """

    _tags = {
        "tag_name": "tests:vm",
        "parent_type": "object",
        "tag_type": "bool",
        "short_descr": "whether tests require their own VM to run",
    }


class tests__skip_by_name(_BaseTag):
    """List of test names to skip when running estimator checks on CI.

    - String name: ``"tests:skip_by_name"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "tests:skip_by_name",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": (
            "list of test names to skip when running estimator checks on CI"
        ),
    }


class tests__python_dependencies(_BaseTag):
    """Additional python dependencies needed in tests, str or list of str (PEP 440).

    - String name: ``"tests:python_dependencies"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "tests:python_dependencies",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": (
            "additional python dependencies needed in tests, "
            "str or list of str (PEP 440)"
        ),
    }


# ------------------
# BaseProbaRegressor
# ------------------


class capability__survival(_BaseTag):
    """Whether estimator can use censoring information, for survival analysis.

    - String name: ``"capability:survival"``
    - Values: ``bool``
    """

    _tags = {
        "tag_name": "capability:survival",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": (
            "whether estimator can use censoring information, "
            "for survival analysis"
        ),
    }


class capability__multioutput(_BaseTag):
    """Whether estimator supports multioutput regression.

    - String name: ``"capability:multioutput"``
    - Values: ``bool``
    """

    _tags = {
        "tag_name": "capability:multioutput",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports multioutput regression",
    }


class capability__missing(_BaseTag):
    """Whether estimator supports missing values.

    - String name: ``"capability:missing"``
    - Values: ``bool``
    """

    _tags = {
        "tag_name": "capability:missing",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports missing values",
    }


class capability__update(_BaseTag):
    """Whether estimator supports online updates via update.

    - String name: ``"capability:update"``
    - Values: ``bool``
    """

    _tags = {
        "tag_name": "capability:update",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports online updates via update",
    }


class X_inner_mtype(_BaseTag):
    """Which machine type(s) is the internal _fit/_predict able to deal with.

    - String name: ``"X_inner_mtype"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "X_inner_mtype",
        "parent_type": "regressor_proba",
        "tag_type": ("list", "str"),
        "short_descr": (
            "which machine type(s) is the internal _fit/_predict "
            "able to deal with?"
        ),
    }


class y_inner_mtype(_BaseTag):
    """Which machine type(s) is the internal _fit/_predict able to deal with.

    - String name: ``"y_inner_mtype"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "y_inner_mtype",
        "parent_type": "regressor_proba",
        "tag_type": ("list", "str"),
        "short_descr": (
            "which machine type(s) is the internal _fit/_predict "
            "able to deal with?"
        ),
    }


class C_inner_mtype(_BaseTag):
    """Which machine type(s) is the internal _fit/_predict able to deal with.

    - String name: ``"C_inner_mtype"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "C_inner_mtype",
        "parent_type": "regressor_proba",
        "tag_type": ("list", "str"),
        "short_descr": (
            "which machine type(s) is the internal _fit/_predict "
            "able to deal with?"
        ),
    }


# ----------------
# BaseDistribution
# ----------------


class capabilities__approx(_BaseTag):
    """Methods of distr that are approximate.

    - String name: ``"capabilities:approx"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "capabilities:approx",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "methods of distr that are approximate",
    }


class capabilities__exact(_BaseTag):
    """Methods of distr that are numerically exact.

    - String name: ``"capabilities:exact"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "capabilities:exact",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "methods of distr that are numerically exact",
    }


class capabilities__undefined(_BaseTag):
    """Methods of distr that are mathematically undefined.

    - String name: ``"capabilities:undefined"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "capabilities:undefined",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "methods of distr that are mathematically undefined",
    }


class distr__measuretype(_BaseTag):
    """Measure type of distr.

    - String name: ``"distr:measuretype"``
    - Values: ``("str", ["continuous", "discrete", "mixed"])``
    """

    _tags = {
        "tag_name": "distr:measuretype",
        "parent_type": "distribution",
        "tag_type": ("str", ["continuous", "discrete", "mixed"]),
        "short_descr": "measure type of distr",
    }


class distr__paramtype(_BaseTag):
    """Parametrization type of distribution.

    - String name: ``"distr:paramtype"``
    - Values: ``("str", ["general", "parametric", "nonparametric", "composite"])``
    """

    _tags = {
        "tag_name": "distr:paramtype",
        "parent_type": "distribution",
        "tag_type": ("str", ["general", "parametric", "nonparametric", "composite"]),
        "short_descr": "parametrization type of distribution",
    }


class approx_mean_spl(_BaseTag):
    """Sample size used in MC estimates of mean.

    - String name: ``"approx_mean_spl"``
    - Values: ``int``
    """

    _tags = {
        "tag_name": "approx_mean_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of mean",
    }


class approx_var_spl(_BaseTag):
    """Sample size used in MC estimates of var.

    - String name: ``"approx_var_spl"``
    - Values: ``int``
    """

    _tags = {
        "tag_name": "approx_var_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of var",
    }


class approx_energy_spl(_BaseTag):
    """Sample size used in MC estimates of energy.

    - String name: ``"approx_energy_spl"``
    - Values: ``int``
    """

    _tags = {
        "tag_name": "approx_energy_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of energy",
    }


class approx_spl(_BaseTag):
    """Sample size used in other MC estimates.

    - String name: ``"approx_spl"``
    - Values: ``int``
    """

    _tags = {
        "tag_name": "approx_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in other MC estimates",
    }


class bisect_iter(_BaseTag):
    """Max iters for bisection method in ppf.

    - String name: ``"bisect_iter"``
    - Values: ``int``
    """

    _tags = {
        "tag_name": "bisect_iter",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "max iters for bisection method in ppf",
    }


class broadcast_params(_BaseTag):
    """Distribution parameters to broadcast, complement is not broadcast.

    - String name: ``"broadcast_params"``
    - Values: ``("list", "str")``
    """

    _tags = {
        "tag_name": "broadcast_params",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": (
            "distribution parameters to broadcast, complement is not broadcast"
        ),
    }


class broadcast_init(_BaseTag):
    """Whether to initialize broadcast parameters in __init__, 'on' or 'off'.

    - String name: ``"broadcast_init"``
    - Values: ``("str", ["on", "off"])``
    """

    _tags = {
        "tag_name": "broadcast_init",
        "parent_type": "distribution",
        "tag_type": ("str", ["on", "off"]),
        "short_descr": (
            "whether to initialize broadcast parameters in __init__, 'on' or 'off'"
        ),
    }


class broadcast_inner(_BaseTag):
    """If inner logic is vectorized ('array') or scalar ('scalar').

    - String name: ``"broadcast_inner"``
    - Values: ``("str", ["array", "scalar"])``
    """

    _tags = {
        "tag_name": "broadcast_inner",
        "parent_type": "distribution",
        "tag_type": ("str", ["array", "scalar"]),
        "short_descr": "if inner logic is vectorized ('array') or scalar ('scalar')",
    }


# ---------------
# BaseProbaMetric
# ---------------


class scitype__y_pred(_BaseTag):
    """Expected input type for y_pred in performance metric.

    - String name: ``"scitype:y_pred"``
    - Values: ``str``
    """

    _tags = {
        "tag_name": "scitype:y_pred",
        "parent_type": "metric",
        "tag_type": "str",
        "short_descr": "expected input type for y_pred in performance metric",
    }


class lower_is_better(_BaseTag):
    """Whether lower (True) or higher (False) is better.

    - String name: ``"lower_is_better"``
    - Values: ``bool``
    """

    _tags = {
        "tag_name": "lower_is_better",
        "parent_type": "metric",
        "tag_type": "bool",
        "short_descr": "whether lower (True) or higher (False) is better",
    }


class capability__survival_metric(_BaseTag):
    """Whether metric uses censoring information, for survival analysis.

    - String name: ``"capability:survival"``
    - Values: ``bool``
    """

    _tags = {
        "tag_name": "capability:survival",
        "parent_type": "metric",
        "tag_type": "bool",
        "short_descr": (
            "whether metric uses censoring information, for survival analysis"
        ),
    }


# ----------------------------
# BaseMetaObject reserved tags
# ----------------------------


class named_object_parameters(_BaseTag):
    """Name of component list attribute for meta-objects.

    - String name: ``"named_object_parameters"``
    - Values: ``str``
    """

    _tags = {
        "tag_name": "named_object_parameters",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "name of component list attribute for meta-objects",
    }


class fitted_named_object_parameters(_BaseTag):
    """Name of fitted component list attribute for meta-objects.

    - String name: ``"fitted_named_object_parameters"``
    - Values: ``str``
    """

    _tags = {
        "tag_name": "fitted_named_object_parameters",
        "parent_type": "estimator",
        "tag_type": "str",
        "short_descr": "name of fitted component list attribute for meta-objects",
    }


# construct the tag register from all classes in this module
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
