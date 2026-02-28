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


import inspect
import sys
import pandas as pd
from skbase.base import BaseObject

# ---------------------------------------------------------
# Tag Class Definitions
# ---------------------------------------------------------

class _BaseTag(BaseObject):
    """Base class for all tags in skpro.
    
    This follows the class-based tag registry pattern for better
    extensibility and alignment with the sktime architecture.
    """
    _tags = {
        "object_type": "tag",
        "tag_name": "",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "",
    }

# --------------------------
# All objects and estimators
# --------------------------

class reserved_params(_BaseTag):
    """List of reserved parameter names."""
    _tags = {
        "tag_name": "reserved_params",
        "parent_type": "object",
        "tag_type": "list",
        "short_descr": "list of reserved parameter names",
    }

class object_type(_BaseTag):
    """Type of object, e.g., 'regressor', 'transformer'."""
    _tags = {
        "tag_name": "object_type",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "type of object, e.g., 'regressor', 'transformer'",
    }

class estimator_type(_BaseTag):
    """Type of estimator, e.g., 'regressor', 'transformer'."""
    _tags = {
        "tag_name": "estimator_type",
        "parent_type": "estimator",
        "tag_type": "str",
        "short_descr": "type of estimator, e.g., 'regressor', 'transformer'",
    }

# Packaging information
# ---------------------

class maintainers(_BaseTag):
    """List of current maintainers of the object."""
    _tags = {
        "tag_name": "maintainers",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of current maintainers of the object, each maintainer a GitHub handle",
    }

class authors(_BaseTag):
    """List of authors of the object."""
    _tags = {
        "tag_name": "authors",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of authors of the object, each author a GitHub handle",
    }

class python_version(_BaseTag):
    """Python version specifier (PEP 440)."""
    _tags = {
        "tag_name": "python_version",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "python version specifier (PEP 440) for estimator, or None = all versions ok",
    }

class python_dependencies(_BaseTag):
    """Python dependencies of estimator."""
    _tags = {
        "tag_name": "python_dependencies",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "python dependencies of estimator as str or list of str",
    }

class python_dependencies_alias(_BaseTag):
    """Import name aliases for dependencies."""
    _tags = {
        "tag_name": "python_dependencies_alias",
        "parent_type": "object",
        "tag_type": "dict",
        "short_descr": "should be provided if import name differs from package name",
    }

class license_type(_BaseTag):
    """License type for interfaced packages."""
    _tags = {
        "tag_name": "license_type",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "license type for interfaced packages: 'copyleft', 'permissive', 'copyright'",
    }

# CI and test flags
# -----------------

class tests_libs(_BaseTag):
    """Library dependencies required for tests."""
    _tags = {
        "tag_name": "tests:libs",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of library dependencies required for tests",
    }

class tests_vm(_BaseTag):
    """Whether tests require their own VM."""
    _tags = {
        "tag_name": "tests:vm",
        "parent_type": "object",
        "tag_type": "bool",
        "short_descr": "whether tests require their own VM to run",
    }

class tests_skip_by_name(_BaseTag):
    """Test names to skip on CI."""
    _tags = {
        "tag_name": "tests:skip_by_name",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "list of test names to skip when running estimator checks on CI",
    }

class tests_python_dependencies(_BaseTag):
    """Additional dependencies for tests."""
    _tags = {
        "tag_name": "tests:python_dependencies",
        "parent_type": "object",
        "tag_type": ("list", "str"),
        "short_descr": "additional python dependencies needed in tests (PEP 440)",
    }

# ------------------
# BaseProbaRegressor
# ------------------

class capability_survival(_BaseTag):
    """Capability for survival analysis."""
    _tags = {
        "tag_name": "capability:survival",
        "parent_type": ["regressor_proba", "metric"],
        "tag_type": "bool",
        "short_descr": "whether estimator can use censoring information, for survival analysis",
    }

class capability_multioutput(_BaseTag):
    """Support for multioutput regression."""
    _tags = {
        "tag_name": "capability:multioutput",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports multioutput regression",
    }

class capability_missing(_BaseTag):
    """Support for missing values."""
    _tags = {
        "tag_name": "capability:missing",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports missing values",
    }

class capability_update(_BaseTag):
    """Support for online updates."""
    _tags = {
        "tag_name": "capability:update",
        "parent_type": "regressor_proba",
        "tag_type": "bool",
        "short_descr": "whether estimator supports online updates via update",
    }

class X_inner_mtype(_BaseTag):
    """Internal X machine type."""
    _tags = {
        "tag_name": "X_inner_mtype",
        "parent_type": "regressor_proba",
        "tag_type": ("list", "str"),
        "short_descr": "which machine type(s) is the internal _fit/_predict able to deal with?",
    }

class y_inner_mtype(_BaseTag):
    """Internal y machine type."""
    _tags = {
        "tag_name": "y_inner_mtype",
        "parent_type": "regressor_proba",
        "tag_type": ("list", "str"),
        "short_descr": "which machine type(s) is the internal _fit/_predict able to deal with?",
    }

class C_inner_mtype(_BaseTag):
    """Internal censoring machine type."""
    _tags = {
        "tag_name": "C_inner_mtype",
        "parent_type": "regressor_proba",
        "tag_type": ("list", "str"),
        "short_descr": "which machine type(s) is the internal _fit/_predict able to deal with?",
    }

# ----------------
# BaseDistribution
# ----------------

class capabilities_approx(_BaseTag):
    """Approximate methods of distribution."""
    _tags = {
        "tag_name": "capabilities:approx",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "methods of distr that are approximate",
    }

class capabilities_exact(_BaseTag):
    """Numerically exact methods of distribution."""
    _tags = {
        "tag_name": "capabilities:exact",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "methods of distr that are numerically exact",
    }

class distr_measuretype(_BaseTag):
    """Measure type of distribution."""
    _tags = {
        "tag_name": "distr:measuretype",
        "parent_type": "distribution",
        "tag_type": ("str", ["continuous", "discrete", "mixed"]),
        "short_descr": "measure type of distr",
    }

class distr_paramtype(_BaseTag):
    """Parametrization type of distribution."""
    _tags = {
        "tag_name": "distr:paramtype",
        "parent_type": "distribution",
        "tag_type": ("str", ["general", "parametric", "nonparametric", "composite"]),
        "short_descr": "parametrization type of distribution",
    }

class approx_mean_spl(_BaseTag):
    """Sample size for MC mean estimates."""
    _tags = {
        "tag_name": "approx_mean_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of mean",
    }

class approx_var_spl(_BaseTag):
    """Sample size for MC variance estimates."""
    _tags = {
        "tag_name": "approx_var_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of var",
    }

class approx_energy_spl(_BaseTag):
    """Sample size for MC energy estimates."""
    _tags = {
        "tag_name": "approx_energy_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in MC estimates of energy",
    }

class approx_spl(_BaseTag):
    """Sample size for other MC estimates."""
    _tags = {
        "tag_name": "approx_spl",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "sample size used in other MC estimates",
    }

class bisect_iter(_BaseTag):
    """Max iterations for bisection method."""
    _tags = {
        "tag_name": "bisect_iter",
        "parent_type": "distribution",
        "tag_type": "int",
        "short_descr": "max iters for bisection method in ppf",
    }

class broadcast_params(_BaseTag):
    """Parameters to broadcast."""
    _tags = {
        "tag_name": "broadcast_params",
        "parent_type": "distribution",
        "tag_type": ("list", "str"),
        "short_descr": "distribution parameters to broadcast",
    }

class broadcast_init(_BaseTag):
    """Whether to initialize broadcast parameters."""
    _tags = {
        "tag_name": "broadcast_init",
        "parent_type": "distribution",
        "tag_type": ("str", ["on", "off"]),
        "short_descr": "whether to initialize broadcast parameters in __init__",
    }

class broadcast_inner(_BaseTag):
    """Inner logic vectorization type."""
    _tags = {
        "tag_name": "broadcast_inner",
        "parent_type": "distribution",
        "tag_type": ("str", ["array", "scalar"]),
        "short_descr": "if inner logic is vectorized ('array') or scalar ('scalar')",
    }

# ---------------
# BaseProbaMetric
# ---------------

class scitype_y_pred(_BaseTag):
    """Expected input type for y_pred."""
    _tags = {
        "tag_name": "scitype:y_pred",
        "parent_type": "metric",
        "tag_type": "str",
        "short_descr": "expected input type for y_pred in performance metric",
    }

class lower_is_better(_BaseTag):
    """Direction of metric optimization."""
    _tags = {
        "tag_name": "lower_is_better",
        "parent_type": "metric",
        "tag_type": "bool",
        "short_descr": "whether lower (True) or higher (False) is better",
    }

# ----------------------------
# BaseMetaObject reserved tags
# ----------------------------

class named_object_parameters(_BaseTag):
    """Component list attribute name."""
    _tags = {
        "tag_name": "named_object_parameters",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "name of component list attribute for meta-objects",
    }

class fitted_named_object_parameters(_BaseTag):
    """Fitted component list attribute name."""
    _tags = {
        "tag_name": "fitted_named_object_parameters",
        "parent_type": "estimator",
        "tag_type": "str",
        "short_descr": "name of fitted component list attribute for meta-objects",
    }

# ---------------------------------------------------------
# Registry Generation Logic
# ---------------------------------------------------------

ESTIMATOR_TAG_REGISTER = []
tag_classes = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for _, cl in tag_classes:
    if cl.__name__ == "_BaseTag" or not issubclass(cl, _BaseTag):
        continue
    
    cl_tags = cl.get_class_tags()
    tag_name = cl_tags.get("tag_name", "unknown_tag")
    parent_type = cl_tags.get("parent_type", "object")
    tag_type = cl_tags.get("tag_type", "str")
    short_descr = cl_tags.get("short_descr", "")
    
    if isinstance(parent_type, list):
        for p_type in parent_type:
            ESTIMATOR_TAG_REGISTER.append((tag_name, p_type, tag_type, short_descr))
    else:
        ESTIMATOR_TAG_REGISTER.append((tag_name, parent_type, tag_type, short_descr))

ESTIMATOR_TAG_TABLE = pd.DataFrame(ESTIMATOR_TAG_REGISTER)
ESTIMATOR_TAG_LIST = ESTIMATOR_TAG_TABLE[0].unique().tolist()

# Compatibility Aliases
OBJECT_TAG_REGISTER = ESTIMATOR_TAG_REGISTER
OBJECT_TAG_TABLE = ESTIMATOR_TAG_TABLE
OBJECT_TAG_LIST = ESTIMATOR_TAG_LIST

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
        raise KeyError(f"{tag_name} is not a valid tag")
    
    tag_row = OBJECT_TAG_TABLE[OBJECT_TAG_TABLE[0] == tag_name]
    tag_type = tag_row.iloc[0, 2]

    # Validation logic for strings/types
    if isinstance(tag_type, str):
        if tag_type == "bool" and not isinstance(tag_value, bool):
            raise ValueError(f"{tag_name} must be bool, found {type(tag_value)}")
        if tag_type == "int" and not isinstance(tag_value, int):
            raise ValueError(f"{tag_name} must be int, found {type(tag_value)}")
        if tag_type == "str" and not isinstance(tag_value, str):
            raise ValueError(f"{tag_name} must be str, found {type(tag_value)}")
        if tag_type == "list" and not isinstance(tag_value, list):
            raise ValueError(f"{tag_name} must be list, found {type(tag_value)}")

    # Validation logic for complex types (tuples)
    elif isinstance(tag_type, tuple):
        if tag_type[0] == "str":
            if tag_value not in tag_type[1]:
                raise ValueError(f"{tag_name} must be one of {tag_type[1]}, found {tag_value}")
        
        elif tag_type[0] == "list" and tag_type[1] == "str":
            if not isinstance(tag_value, (str, list)):
                 raise ValueError(f"{tag_name} must be str or list of str, found {type(tag_value)}")
            if isinstance(tag_value, list) and not all(isinstance(x, str) for x in tag_value):
                 raise ValueError(f"{tag_name} must be a list of strings.")

        elif tag_type[0] == "list" and isinstance(tag_type[1], list):
            if not isinstance(tag_value, list) or not set(tag_value).issubset(tag_type[1]):
                raise ValueError(f"{tag_name} must be a subset of {tag_type[1]}")