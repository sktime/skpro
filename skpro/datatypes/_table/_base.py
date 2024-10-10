# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Base class for data types."""

__author__ = ["fkiraly"]

from skpro.datatypes._base import BaseDatatype


class BaseTable(BaseDatatype):
    """Base class for Table data types.

    Parameters are inferred by check.

    Parameters
    ----------
    is_univariate: bool
        True iff table has one variable
    is_empty: bool
        True iff table has no variables or no instances
    has_nans: bool
        True iff the table contains NaN values
    n_instances: int
        number of instances/rows in the table
    n_features: int
        number of variables in table
    feature_names: list of int or object
        names of variables in table
    """

    _tags = {
        "scitype": "Table",
        "name": None,  # any string
        "name_python": None,  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": None,
    }

    def __init__(
        self,
        is_univariate=None,
        is_empty=None,
        has_nans=None,
        n_instances=None,
        n_features=None,
        feature_names=None,
    ):
        self.is_univariate = is_univariate
        self.is_empty = is_empty
        self.has_nans = has_nans
        self.n_instances = n_instances
        self.n_features = n_features
        self.feature_names = feature_names

        super().__init__()
