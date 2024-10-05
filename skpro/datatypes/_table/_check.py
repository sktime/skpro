"""Machine type classes for Table scitype.

Checks for each class are defined in the "check" method, of signature:

Parameters
----------
obj - object to check
return_metadata - bool, optional, default=False
    if False, returns only "valid" return
    if True, returns all three return objects
var_name: str, optional, default="obj" - name of input in error messages

Returns
-------
valid: bool - whether obj is a valid object of mtype/scitype
msg: str - error message if object is not valid, otherwise None
        returned only if return_metadata is True
metadata: dict - metadata about obj if valid, otherwise None
        returned only if return_metadata is True
    fields:
        "is_univariate": bool, True iff table has one variable
        "is_empty": bool, True iff table has no variables or no instances
        "has_nans": bool, True iff the panel contains NaN values
        "n_instances": int, number of instances/rows in the table
        "n_features": int, number of variables in table
        "feature_names": list of int or object, names of variables in table
"""

__author__ = ["fkiraly"]

import numpy as np
import pandas as pd

from skpro.datatypes._common import _req, _ret
from skpro.datatypes._table._base import BaseTable

PRIMITIVE_TYPES = (float, int, str)


class TablePdDataFrame(BaseTable):
    """Data type: pandas.DataFrame based specification of data frame table.

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
        "name": "pd_DataFrame_Table",  # any string
        "name_python": "table_pd_df",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "pandas",
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:index": True,
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
        super().__init__(
            is_univariate=is_univariate,
            n_instances=n_instances,
            is_empty=is_empty,
            has_nans=has_nans,
            n_features=n_features,
            feature_names=feature_names,
        )

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        return _check_pddataframe_table(obj, return_metadata, var_name)


def _check_pddataframe_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, pd.DataFrame):
        msg = f"{var_name} must be a pandas.DataFrame, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a pd.DataFrame
    index = obj.index
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1 or len(obj.columns) < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = len(obj.columns) < 2
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(index)
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isna().values.any()
    if _req("n_features", return_metadata):
        metadata["n_features"] = len(obj.columns)
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = obj.columns.to_list()

    # check that no dtype is object
    if "object" in obj.dtypes.values:
        msg = f"{var_name} should not have column of 'object' dtype"
        return _ret(False, msg, None, return_metadata)

    return _ret(True, None, metadata, return_metadata)


class TablePdSeries(BaseTable):
    """Data type: pandas.Series based specification of data frame table.

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
        "name": "pd_Series_Table",  # any string
        "name_python": "table_pd_series",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "pandas",
        "capability:multivariate": False,
        "capability:missing_values": True,
        "capability:index": True,
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
        super().__init__(
            is_univariate=is_univariate,
            n_instances=n_instances,
            is_empty=is_empty,
            has_nans=has_nans,
            n_features=n_features,
            feature_names=feature_names,
        )

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        return _check_pdseries_table(obj, return_metadata, var_name)


def _check_pdseries_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, pd.Series):
        msg = f"{var_name} must be a pandas.Series, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a pd.Series
    index = obj.index
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(index) < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = True
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(index)
    if _req("n_features", return_metadata):
        metadata["n_features"] = 1
    if _req("feature_names", return_metadata):
        if not hasattr(obj, "name") or obj.name is None:
            metadata["feature_names"] = [0]
        else:
            metadata["feature_names"] = [obj.name]

    # check that dtype is not object
    if "object" == obj.dtypes:
        msg = f"{var_name} should not be of 'object' dtype"
        return _ret(False, msg, None, return_metadata)

    # check whether index is equally spaced or if there are any nans
    #   compute only if needed
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = obj.isna().values.any()

    return _ret(True, None, metadata, return_metadata)


class TableNp1D(BaseTable):
    """Data type: 1D np.ndarray based specification of data frame table.

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
        "name": "numpy1D",  # any string
        "name_python": "table_numpy1d",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "numpy",
        "capability:multivariate": False,
        "capability:missing_values": True,
        "capability:index": False,
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
        super().__init__(
            is_univariate=is_univariate,
            n_instances=n_instances,
            is_empty=is_empty,
            has_nans=has_nans,
            n_features=n_features,
            feature_names=feature_names,
        )

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        return _check_numpy1d_table(obj, return_metadata, var_name)


def _check_numpy1d_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if len(obj.shape) != 1:
        msg = f"{var_name} must be 1D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 1D np.ndarray
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(obj)
    # 1D numpy arrays are considered univariate
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = True
    # check whether there any nans; compute only if requested
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = pd.isnull(obj).any()
    # 1D numpy arrays are considered univariate, with one feature named 0 (integer)
    if _req("n_features", return_metadata):
        metadata["n_features"] = 1
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = [0]

    return _ret(True, None, metadata, return_metadata)


class TableNp2D(BaseTable):
    """Data type: 2D np.ndarray based specification of data frame table.

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
        "name": "numpy2D",  # any string
        "name_python": "table_numpy2d",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "numpy",
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:index": False,
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
        super().__init__(
            is_univariate=is_univariate,
            n_instances=n_instances,
            is_empty=is_empty,
            has_nans=has_nans,
            n_features=n_features,
            feature_names=feature_names,
        )

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        return _check_numpy2d_table(obj, return_metadata, var_name)


def _check_numpy2d_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, np.ndarray):
        msg = f"{var_name} must be a numpy.ndarray, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if len(obj.shape) != 2:
        msg = f"{var_name} must be 1D or 2D numpy.ndarray, but found {len(obj.shape)}D"
        return _ret(False, msg, None, return_metadata)

    # we now know obj is a 2D np.ndarray
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1 or obj.shape[1] < 1
    if _req("is_univariate", return_metadata):
        metadata["is_univariate"] = obj.shape[1] < 2
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = obj.shape[0]
    # check whether there any nans; compute only if requested
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = pd.isnull(obj).any()
    # 1D numpy arrays are considered univariate, with integer feature names
    if _req("n_features", return_metadata):
        metadata["n_features"] = obj.shape[1]
    if _req("feature_names", return_metadata):
        metadata["feature_names"] = list(range(obj.shape[1]))

    return _ret(True, None, metadata, return_metadata)


class TableListOfDict(BaseTable):
    """Data type: list of dict based specification of data frame table.

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
        "name": "list_of_dict",  # any string
        "name_python": "table_list_of_dict",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": "numpy",
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:index": False,
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
        super().__init__(
            is_univariate=is_univariate,
            n_instances=n_instances,
            is_empty=is_empty,
            has_nans=has_nans,
            n_features=n_features,
            feature_names=feature_names,
        )

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        return _check_list_of_dict_table(obj, return_metadata, var_name)


def _check_list_of_dict_table(obj, return_metadata=False, var_name="obj"):
    metadata = dict()

    if not isinstance(obj, list):
        msg = f"{var_name} must be a list of dict, found {type(obj)}"
        return _ret(False, msg, None, return_metadata)

    if not np.all([isinstance(x, dict) for x in obj]):
        msg = (
            f"{var_name} must be a list of dict, but elements at following "
            f"indices are not dict: {np.where(not isinstance(x, dict) for x in obj)}"
        )
        return _ret(False, msg, None, return_metadata)

    for i, d in enumerate(obj):
        for key in d.keys():
            if not isinstance(d[key], PRIMITIVE_TYPES):
                msg = (
                    "all entries must be of primitive type (str, int, float), but "
                    f"found {type(d[key])} at index {i}, key {key}"
                )

    # we now know obj is a list of dict
    # check whether there any nans; compute only if requested
    if _req("is_univariate", return_metadata):
        multivariate_because_one_row = np.any([len(x) > 1 for x in obj])
        if not multivariate_because_one_row:
            all_keys = np.unique([key for d in obj for key in d.keys()])
            multivariate_because_keys_different = len(all_keys) > 1
            multivariate = multivariate_because_keys_different
        else:
            multivariate = multivariate_because_one_row
        metadata["is_univariate"] = not multivariate
    if _req("has_nans", return_metadata):
        metadata["has_nans"] = np.any(
            [pd.isnull(d[key]) for d in obj for key in d.keys()]
        )
    if _req("is_empty", return_metadata):
        metadata["is_empty"] = len(obj) < 1 or np.all([len(x) < 1 for x in obj])
    if _req("n_instances", return_metadata):
        metadata["n_instances"] = len(obj)

    # this can be expensive, so compute only if requested
    if _req("n_features", return_metadata) or _req("feature_names", return_metadata):
        all_keys = np.unique([key for d in obj for key in d.keys()])
        if _req("n_features", return_metadata):
            metadata["n_features"] = len(all_keys)
        if _req("feature_names", return_metadata):
            metadata["feature_names"] = all_keys.tolist()

    return _ret(True, None, metadata, return_metadata)


class TablePolarsEager(BaseTable):
    """Data type: eager polars DataFrame based specification of data frame table.

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
        "name": "polars_eager_table",  # any string
        "name_python": "table_polars_eager",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": ["polars", "pyarrow"],
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:index": False,
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
        super().__init__(
            is_univariate=is_univariate,
            n_instances=n_instances,
            is_empty=is_empty,
            has_nans=has_nans,
            n_features=n_features,
            feature_names=feature_names,
        )

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        from skpro.datatypes._adapter.polars import check_polars_frame

        return check_polars_frame(obj, return_metadata, var_name, lazy=False)


class TablePolarsLazy(BaseTable):
    """Data type: lazy polars DataFrame based specification of data frame table.

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
        "name": "polars_lazy_table",  # any string
        "name_python": "table_polars_lazy",  # lower_snake_case
        "name_aliases": [],
        "python_version": None,
        "python_dependencies": ["polars", "pyarrow"],
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:index": False,
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
        super().__init__(
            is_univariate=is_univariate,
            n_instances=n_instances,
            is_empty=is_empty,
            has_nans=has_nans,
            n_features=n_features,
            feature_names=feature_names,
        )

    def _check(self, obj, return_metadata=False, var_name="obj"):
        """Check if obj is of this data type.

        Parameters
        ----------
        obj : any
            Object to check.
        return_metadata : bool, optional (default=False)
            Whether to return metadata.
        var_name : str, optional (default="obj")
            Name of the variable to check, for use in error messages.

        Returns
        -------
        valid : bool
            Whether obj is of this data type.
        msg : str, only returned if return_metadata is True.
            Error message if obj is not of this data type.
        metadata : dict, only returned if return_metadata is True.
            Metadata dictionary.
        """
        from skpro.datatypes._adapter.polars import check_polars_frame

        return check_polars_frame(obj, return_metadata, var_name, lazy=True)
