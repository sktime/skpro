"""Register of estimator and object tags.

Note for extenders: new tags should be entered in OBJECT_TAG_REGISTER.
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
import pandas as pd

OBJECT_TAG_REGISTER = [
    # --------------------------
    # all objects and estimators
    # --------------------------
    (
        "reserved_params",
        "object",
        "list",
        "list of reserved parameter names",
    ),
    (
        "object_type",
        "object",
        "str",
        "type of object, e.g., 'regressor', 'transformer'",
    ),
    (
        "estimator_type",
        "estimator",
        "str",
        "type of estimator, e.g., 'regressor', 'transformer'",
    ),
    # packaging information
    # ---------------------
    (
        "maintainers",
        "object",
        ("list", "str"),
        "list of current maintainers of the object, each maintainer a GitHub handle",
    ),
    (
        "authors",
        "object",
        ("list", "str"),
        "list of authors of the object, each author a GitHub handle",
    ),
    (
        "python_version",
        "object",
        "str",
        "python version specifier (PEP 440) for estimator, or None = all versions ok",
    ),
    (
        "python_dependencies",
        "object",
        ("list", "str"),
        "python dependencies of estimator as str or list of str",
    ),
    (
        "python_dependencies_alias",
        "object",
        "dict",
        "should be provided if import name differs from package name, \
        key-value pairs are package name, import name",
    ),
    # ------------------
    # BaseProbaRegressor
    # ------------------
    (
        "capability:survival",
        "regressor_proba",
        "bool",
        "whether estimator can use censoring information, for survival analysis",
    ),
    (
        "capability:multioutput",
        "regressor_proba",
        "bool",
        "whether estimator supports multioutput regression",
    ),
    (
        "capability:missing",
        "regressor_proba",
        "bool",
        "whether estimator supports missing values",
    ),
    (
        "X_inner_mtype",
        "regressor_proba",
        ("list", "str"),
        "which machine type(s) is the internal _fit/_predict able to deal with?",
    ),
    (
        "y_inner_mtype",
        "regressor_proba",
        ("list", "str"),
        "which machine type(s) is the internal _fit/_predict able to deal with?",
    ),
    (
        "C_inner_mtype",
        "regressor_proba",
        ("list", "str"),
        "which machine type(s) is the internal _fit/_predict able to deal with?",
    ),
    # ----------------
    # BaseDistribution
    # ----------------
    (
        "capabilities:approx",
        "distribution",
        ("list", "str"),
        "methods of distr that are approximate",
    ),
    (
        "capabilities:exact",
        "distribution",
        ("list", "str"),
        "methods of distr that are numerically exact",
    ),
    (
        "distr:measuretype",
        "distribution",
        ("str", ["continuous", "discrete", "mixed"]),
        "measure type of distr",
    ),
    (
        "approx_mean_spl",
        "distribution",
        "int",
        "sample size used in MC estimates of mean",
    ),
    (
        "approx_var_spl",
        "distribution",
        "int",
        "sample size used in MC estimates of var",
    ),
    (
        "approx_energy_spl",
        "distribution",
        "int",
        "sample size used in MC estimates of energy",
    ),
    (
        "approx_spl",
        "distribution",
        "int",
        "sample size used in other MC estimates",
    ),
    (
        "bisect_iter",
        "distribution",
        "int",
        "max iters for bisection method in ppf",
    ),
    # ---------------
    # BaseProbaMetric
    # ---------------
    (
        "scitype:y_pred",
        "metric",
        "str",
        "expected input type for y_pred in performance metric",
    ),
    (
        "lower_is_better",
        "metric",
        "bool",
        "whether lower (True) or higher (False) is better",
    ),
    (
        "capability:survival",
        "metric",
        "bool",
        "whether metric uses censoring information, for survival analysis",
    ),
    # ----------------------------
    # BaseMetaObject reserved tags
    # ----------------------------
    (
        "named_object_parameters",
        "object",
        "str",
        "name of component list attribute for meta-objects",
    ),
    (
        "fitted_named_object_parameters",
        "estimator",
        "str",
        "name of fitted component list attribute for meta-objects",
    ),
]

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
