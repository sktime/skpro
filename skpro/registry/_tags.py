"""Register of estimator and object tags.

Note for extenders: new tags should be added by defining a new class 
inheriting from _BaseTag in this module. 

The OBJECT_TAG_REGISTER is automatically populated from these classes 
via inspection, so no manual entry into a list is required. 🏗️

This module exports the following:

---
OBJECT_TAG_REGISTER - list of tuples
    The legacy-compatible list of tag metadata. Generated dynamically 
    from _BaseTag subclasses to ensure no manual errors in the list.

---
OBJECT_TAG_TABLE - pd.DataFrame
    The registry in a searchable DataFrame format. 📊

OBJECT_TAG_LIST - list of string
    A simple list of all registered tag names. 📋

---
check_tag_is_valid(tag_name, tag_value)
    A robust validation function that uses the metadata defined in the 
    tag classes to verify user-provided values.
"""

import inspect
import sys

import pandas as pd
from skbase.base import BaseObject

class _BaseTag(BaseObject):
    """Base class for all tags."""
    _tags = {
        "object_type": "tag",
        "tag_name": "fill_this_in",
        "parent_type": "object",
        "tag_type": "str",
        "short_descr": "describe the tag here",
        "user_facing": True,
    }

class reserved_params(_BaseTag):
    """list of reserved parameter names"""
    _tags = {
        "tag_name": 'reserved_params',
        "parent_type": 'object',
        "tag_type": 'list',
        "short_descr": 'list of reserved parameter names',
    }

class object_type(_BaseTag):
    """type of object, e.g., 'regressor', 'transformer'"""
    _tags = {
        "tag_name": 'object_type',
        "parent_type": 'object',
        "tag_type": 'str',
        "short_descr": "type of object, e.g., 'regressor', 'transformer'",
    }

class estimator_type(_BaseTag):
    """type of estimator, e.g., 'regressor', 'transformer'"""
    _tags = {
        "tag_name": 'estimator_type',
        "parent_type": 'estimator',
        "tag_type": 'str',
        "short_descr": "type of estimator, e.g., 'regressor', 'transformer'",
    }

class maintainers(_BaseTag):
    """list of current maintainers of the object, each maintainer a GitHub handle"""
    _tags = {
        "tag_name": 'maintainers',
        "parent_type": 'object',
        "tag_type": ('list', 'str'),
        "short_descr": 'list of current maintainers of the object, each maintainer a GitHub handle',
    }

class authors(_BaseTag):
    """list of authors of the object, each author a GitHub handle"""
    _tags = {
        "tag_name": 'authors',
        "parent_type": 'object',
        "tag_type": ('list', 'str'),
        "short_descr": 'list of authors of the object, each author a GitHub handle',
    }

class python_version(_BaseTag):
    """python version specifier (PEP 440) for estimator, or None = all versions ok"""
    _tags = {
        "tag_name": 'python_version',
        "parent_type": 'object',
        "tag_type": 'str',
        "short_descr": 'python version specifier (PEP 440) for estimator, or None = all versions ok',
    }

class python_dependencies(_BaseTag):
    """python dependencies of estimator as str or list of str"""
    _tags = {
        "tag_name": 'python_dependencies',
        "parent_type": 'object',
        "tag_type": ('list', 'str'),
        "short_descr": 'python dependencies of estimator as str or list of str',
    }

class python_dependencies_alias(_BaseTag):
    """should be provided if import name differs from package name, key-value pairs are package name, import name"""
    _tags = {
        "tag_name": 'python_dependencies_alias',
        "parent_type": 'object',
        "tag_type": 'dict',
        "short_descr": 'should be provided if import name differs from package name, key-value pairs are package name, import name',
    }

class license_type(_BaseTag):
    """license type for interfaced packages: 'copyleft', 'permissive', 'copyright'. may be incorrect, NO LIABILITY assumed for this field"""
    _tags = {
        "tag_name": 'license_type',
        "parent_type": 'object',
        "tag_type": 'str',
        "short_descr": "license type for interfaced packages: 'copyleft', 'permissive', 'copyright'. may be incorrect, NO LIABILITY assumed for this field",
    }

class tests__libs(_BaseTag):
    """list of library dependencies required for tests"""
    _tags = {
        "tag_name": 'tests:libs',
        "parent_type": 'object',
        "tag_type": ('list', 'str'),
        "short_descr": 'list of library dependencies required for tests',
    }

class tests__vm(_BaseTag):
    """whether tests require their own VM to run"""
    _tags = {
        "tag_name": 'tests:vm',
        "parent_type": 'object',
        "tag_type": 'bool',
        "short_descr": 'whether tests require their own VM to run',
    }

class tests__skip_by_name(_BaseTag):
    """list of test names to skip when running estimator checks on CI"""
    _tags = {
        "tag_name": 'tests:skip_by_name',
        "parent_type": 'object',
        "tag_type": ('list', 'str'),
        "short_descr": 'list of test names to skip when running estimator checks on CI',
    }

class tests__python_dependencies(_BaseTag):
    """additional python dependencies needed in tests, str or list of str (PEP 440)"""
    _tags = {
        "tag_name": 'tests:python_dependencies',
        "parent_type": 'object',
        "tag_type": ('list', 'str'),
        "short_descr": 'additional python dependencies needed in tests, str or list of str (PEP 440)',
    }

class capability__survival(_BaseTag):
    """whether estimator can use censoring information, for survival analysis"""
    _tags = {
        "tag_name": 'capability:survival',
        "parent_type": ['regressor_proba', 'metric'],
        "tag_type": 'bool',
        "short_descr": 'whether estimator can use censoring information, for survival analysis',
    }

class capability__multioutput(_BaseTag):
    """whether estimator supports multioutput regression"""
    _tags = {
        "tag_name": 'capability:multioutput',
        "parent_type": 'regressor_proba',
        "tag_type": 'bool',
        "short_descr": 'whether estimator supports multioutput regression',
    }

class capability__missing(_BaseTag):
    """whether estimator supports missing values"""
    _tags = {
        "tag_name": 'capability:missing',
        "parent_type": 'regressor_proba',
        "tag_type": 'bool',
        "short_descr": 'whether estimator supports missing values',
    }

class capability__update(_BaseTag):
    """whether estimator supports online updates via update"""
    _tags = {
        "tag_name": 'capability:update',
        "parent_type": 'regressor_proba',
        "tag_type": 'bool',
        "short_descr": 'whether estimator supports online updates via update',
    }

class X_inner_mtype(_BaseTag):
    """which machine type(s) is the internal _fit/_predict able to deal with?"""
    _tags = {
        "tag_name": 'X_inner_mtype',
        "parent_type": 'regressor_proba',
        "tag_type": ('list', 'str'),
        "short_descr": 'which machine type(s) is the internal _fit/_predict able to deal with?',
    }

class y_inner_mtype(_BaseTag):
    """which machine type(s) is the internal _fit/_predict able to deal with?"""
    _tags = {
        "tag_name": 'y_inner_mtype',
        "parent_type": 'regressor_proba',
        "tag_type": ('list', 'str'),
        "short_descr": 'which machine type(s) is the internal _fit/_predict able to deal with?',
    }

class C_inner_mtype(_BaseTag):
    """which machine type(s) is the internal _fit/_predict able to deal with?"""
    _tags = {
        "tag_name": 'C_inner_mtype',
        "parent_type": 'regressor_proba',
        "tag_type": ('list', 'str'),
        "short_descr": 'which machine type(s) is the internal _fit/_predict able to deal with?',
    }

class capabilities__approx(_BaseTag):
    """methods of distr that are approximate"""
    _tags = {
        "tag_name": 'capabilities:approx',
        "parent_type": 'distribution',
        "tag_type": ('list', 'str'),
        "short_descr": 'methods of distr that are approximate',
    }

class capabilities__exact(_BaseTag):
    """methods of distr that are numerically exact"""
    _tags = {
        "tag_name": 'capabilities:exact',
        "parent_type": 'distribution',
        "tag_type": ('list', 'str'),
        "short_descr": 'methods of distr that are numerically exact',
    }

class distr__measuretype(_BaseTag):
    """measure type of distr"""
    _tags = {
        "tag_name": 'distr:measuretype',
        "parent_type": 'distribution',
        "tag_type": ('str', ['continuous', 'discrete', 'mixed']),
        "short_descr": 'measure type of distr',
    }

class distr__paramtype(_BaseTag):
    """parametrization type of distribution"""
    _tags = {
        "tag_name": 'distr:paramtype',
        "parent_type": 'distribution',
        "tag_type": ('str', ['general', 'parametric', 'nonparametric', 'composite']),
        "short_descr": 'parametrization type of distribution',
    }

class approx_mean_spl(_BaseTag):
    """sample size used in MC estimates of mean"""
    _tags = {
        "tag_name": 'approx_mean_spl',
        "parent_type": 'distribution',
        "tag_type": 'int',
        "short_descr": 'sample size used in MC estimates of mean',
    }

class approx_var_spl(_BaseTag):
    """sample size used in MC estimates of var"""
    _tags = {
        "tag_name": 'approx_var_spl',
        "parent_type": 'distribution',
        "tag_type": 'int',
        "short_descr": 'sample size used in MC estimates of var',
    }

class approx_energy_spl(_BaseTag):
    """sample size used in MC estimates of energy"""
    _tags = {
        "tag_name": 'approx_energy_spl',
        "parent_type": 'distribution',
        "tag_type": 'int',
        "short_descr": 'sample size used in MC estimates of energy',
    }

class approx_spl(_BaseTag):
    """sample size used in other MC estimates"""
    _tags = {
        "tag_name": 'approx_spl',
        "parent_type": 'distribution',
        "tag_type": 'int',
        "short_descr": 'sample size used in other MC estimates',
    }

class bisect_iter(_BaseTag):
    """max iters for bisection method in ppf"""
    _tags = {
        "tag_name": 'bisect_iter',
        "parent_type": 'distribution',
        "tag_type": 'int',
        "short_descr": 'max iters for bisection method in ppf',
    }

class broadcast_params(_BaseTag):
    """distribution parameters to broadcast, complement is not broadcast"""
    _tags = {
        "tag_name": 'broadcast_params',
        "parent_type": 'distribution',
        "tag_type": ('list', 'str'),
        "short_descr": 'distribution parameters to broadcast, complement is not broadcast',
    }

class broadcast_init(_BaseTag):
    """whether to initialize broadcast parameters in __init__, 'on' or 'off'"""
    _tags = {
        "tag_name": 'broadcast_init',
        "parent_type": 'distribution',
        "tag_type": ('str', ['on', 'off']),
        "short_descr": "whether to initialize broadcast parameters in __init__, 'on' or 'off'",
    }

class broadcast_inner(_BaseTag):
    """if inner logic is vectorized ('array') or scalar ('scalar')"""
    _tags = {
        "tag_name": 'broadcast_inner',
        "parent_type": 'distribution',
        "tag_type": ('str', ['array', 'scalar']),
        "short_descr": "if inner logic is vectorized ('array') or scalar ('scalar')",
    }

class scitype__y_pred(_BaseTag):
    """expected input type for y_pred in performance metric"""
    _tags = {
        "tag_name": 'scitype:y_pred',
        "parent_type": 'metric',
        "tag_type": 'str',
        "short_descr": 'expected input type for y_pred in performance metric',
    }

class lower_is_better(_BaseTag):
    """whether lower (True) or higher (False) is better"""
    _tags = {
        "tag_name": 'lower_is_better',
        "parent_type": 'metric',
        "tag_type": 'bool',
        "short_descr": 'whether lower (True) or higher (False) is better',
    }

class named_object_parameters(_BaseTag):
    """name of component list attribute for meta-objects"""
    _tags = {
        "tag_name": 'named_object_parameters',
        "parent_type": 'object',
        "tag_type": 'str',
        "short_descr": 'name of component list attribute for meta-objects',
    }

class fitted_named_object_parameters(_BaseTag):
    """name of fitted component list attribute for meta-objects"""
    _tags = {
        "tag_name": 'fitted_named_object_parameters',
        "parent_type": 'estimator',
        "tag_type": 'str',
        "short_descr": 'name of fitted component list attribute for meta-objects',
    }




# --- Registry Generation ---
ESTIMATOR_TAG_REGISTER = []
tag_clses = inspect.getmembers(sys.modules[__name__], inspect.isclass)

for _, cl in tag_clses:
    if cl.__name__ == "_BaseTag" or not issubclass(cl, _BaseTag):
        continue
    
    cl_tags = cl.get_class_tags()
    tag_name = cl_tags.get("tag_name", "unknown_tag")
    parent_type = cl_tags.get("parent_type", "object")
    tag_type = cl_tags.get("tag_type", "str")
    short_descr = cl_tags.get("short_descr", "")
    
    # Logic to handle single or multiple parent types 🏗️
    if isinstance(parent_type, list):
        for p_type in parent_type:
            ESTIMATOR_TAG_REGISTER.append((tag_name, p_type, tag_type, short_descr))
    else:
        ESTIMATOR_TAG_REGISTER.append((tag_name, parent_type, tag_type, short_descr))

# --- Public API Exports ---
# These are the variables other modules will import
ESTIMATOR_TAG_TABLE = pd.DataFrame(ESTIMATOR_TAG_REGISTER)
ESTIMATOR_TAG_LIST = ESTIMATOR_TAG_TABLE[0].unique().tolist()

# Legacy & Future Compatibility Aliases
OBJECT_TAG_REGISTER = ESTIMATOR_TAG_REGISTER
OBJECT_TAG_TABLE = ESTIMATOR_TAG_TABLE
OBJECT_TAG_LIST = ESTIMATOR_TAG_LIST

def check_tag_is_valid(tag_name, tag_value):
    if tag_name not in OBJECT_TAG_LIST:
        raise KeyError(f"{tag_name} is not a valid tag")
    
    # Safely get the expected type
    tag_row = OBJECT_TAG_TABLE[OBJECT_TAG_TABLE[0] == tag_name]
    tag_type = tag_row.iloc[0, 2]

    # 1. Handle Simple Types (Strings)
    if isinstance(tag_type, str):
        if tag_type == "bool" and not isinstance(tag_value, bool):
            raise ValueError(f"{tag_name} must be bool, found {type(tag_value)}")
        if tag_type == "int":
            if not isinstance(tag_value, int):
                raise ValueError(f"{tag_name} must be int, found {type(tag_value)}")
            if tag_name == "some_positive_tag" and tag_value < 0:
                raise ValueError(f"{tag_name} must be positive, found {tag_value}")
        if tag_type == "str" and not isinstance(tag_value, str):
            raise ValueError(f"{tag_name} must be str, found {type(tag_value)}")
        if tag_type == "list" and not isinstance(tag_value, list):
            raise ValueError(f"{tag_name} must be list, found {type(tag_value)}")

    # 2. Handle Complex Types (Tuples)
    elif isinstance(tag_type, tuple):
        # Case: ("str", ["option1", "option2"])
        if tag_type[0] == "str":
            if tag_value not in tag_type[1]:
                raise ValueError(f"{tag_name} must be one of {tag_type[1]}, found {tag_value}")
        
        # Case: ("list", "str") -> str or list of str
        elif tag_type[0] == "list" and tag_type[1] == "str":
            if not isinstance(tag_value, (str, list)):
                 raise ValueError(f"{tag_name} must be str or list of str, found {type(tag_value)}")
            if isinstance(tag_value, list) and not all(isinstance(x, str) for x in tag_value):
                 raise ValueError(f"{tag_name} must be a list of strings, but contained other types.")

        # Case: ("list", ["allowed1", "allowed2"]) -> subset check
        elif tag_type[0] == "list" and isinstance(tag_type[1], list):
            if not isinstance(tag_value, list):
                raise ValueError(f"{tag_name} must be a list, found {type(tag_value)}")
            if not set(tag_value).issubset(tag_type[1]):
                raise ValueError(f"{tag_name} must be a subset of {tag_type[1]}, found {tag_value}")
