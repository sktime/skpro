"""Module exports: data type definitions, checks, validation, fixtures, converters."""
# this module has been adapted from sktime
# it is largely copy-pasting the Proba and Table parts
# todo: factor this out into a common base

__author__ = ["fkiraly"]

from skpro.datatypes._check import (
    check_is_error_msg,
    check_is_mtype,
    check_is_scitype,
    check_raise,
    mtype,
    scitype,
)
from skpro.datatypes._convert import convert, convert_to
from skpro.datatypes._examples import get_examples
from skpro.datatypes._registry import (
    MTYPE_LIST_PROBA,
    MTYPE_LIST_TABLE,
    MTYPE_REGISTER,
    SCITYPE_LIST,
    SCITYPE_REGISTER,
    mtype_to_scitype,
    scitype_to_mtype,
)

__all__ = [
    "check_is_error_msg",
    "check_is_mtype",
    "check_is_scitype",
    "check_raise",
    "convert",
    "convert_to",
    "mtype",
    "get_examples",
    "mtype_to_scitype",
    "MTYPE_REGISTER",
    "MTYPE_LIST_PROBA",
    "MTYPE_LIST_TABLE",
    "scitype",
    "scitype_to_mtype",
    "SCITYPE_LIST",
    "SCITYPE_REGISTER",
]
