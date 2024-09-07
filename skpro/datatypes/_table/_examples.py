"""Example generation for testing.

Exports dict of examples, useful for testing as fixtures.

example_dict: dict indexed by triple
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are data objects, considered examples for the mtype
    all examples with same index are considered "same" on scitype content
    if None, indicates that representation is not possible

example_lossy: dict of bool indexed by pairs of str
  1st element = mtype - str
  2nd element = considered as this scitype - str
  3rd element = int - index of example
elements are bool, indicate whether representation has information removed
    all examples with same index are considered "same" on scitype content

overall, conversions from non-lossy representations to any other ones
    should yield the element exactly, identidally (given same index)
"""

import numpy as np
import pandas as pd

from skpro.datatypes._base import BaseExample

example_dict = dict()
example_dict_lossy = dict()
example_dict_metadata = dict()

###
# example 0: univariate

class UnivTable(BaseExample):

    _tags = {
        "scitype": "Table",
        "index": 0,
        "metadata": {
            "is_univariate": True,
            "is_empty": False,
            "has_nans": False,
            "n_instances": 4,
            "n_features": 1,
            "feature_names": ["a"],
        },
    }


class UnivTableDf(UnivTable):

    _tags = {
        "mtype": "pd_DataFrame_Table",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return pd.DataFrame({"a": [1, 4, 0.5, -3]})


class UnivTableNumpy2D(UnivTable):

    _tags = {
        "mtype": "numpy2D",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[1], [4], [0.5], [-3]])


class UnivTableNumpy1D(UnivTable):

    _tags = {
        "mtype": "numpy1D",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([1, 4, 0.5, -3])


class UnivTableSeries(UnivTable):

    _tags = {
        "mtype": "pd_Series_Table",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return pd.Series([1, 4, 0.5, -3])


class UnivTableListOfDict(UnivTable):

    _tags = {
        "mtype": "list_of_dict",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return [{"a": 1.0}, {"a": 4.0}, {"a": 0.5}, {"a": -3.0}]


class UnivTablePolarsEager(UnivTable):

    _tags = {
        "mtype": "polars_eager_table",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from skpro.datatypes._adapter.polars import convert_pandas_to_polars_with_index

        df = pd.DataFrame({"a": [1, 4, 0.5, -3]})
        return convert_pandas_to_polars_with_index(df)


class UnivTablePolarsLazy(UnivTable):

    _tags = {
        "mtype": "polars_lazy_table",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from skpro.datatypes._adapter.polars import convert_pandas_to_polars_with_index

        df = pd.DataFrame({"a": [1, 4, 0.5, -3]})
        return convert_pandas_to_polars_with_index(df, lazy=True)


###
# example 1: multivariate

class MultivTable(BaseExample):

    _tags = {
        "scitype": "Table",
        "index": 1,
        "metadata": {
            "is_univariate": False,
            "is_empty": False,
            "has_nans": False,
            "n_instances": 4,
            "n_features": 2,
            "feature_names": ["a", "b"],
        },
    }


class MultivTableDf(MultivTable):

    _tags = {
        "mtype": "pd_DataFrame_Table",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})


class MultivTableNumpy2D(MultivTable):

    _tags = {
        "mtype": "numpy2D",
        "python_dependencies": None,
        "lossy": True,
    }

    def build(self):
        return np.array([[1, 3], [4, 7], [0.5, 2], [-3, -3 / 7]])


class MultivTableNumpy1D(MultivTable):
    
        _tags = {
            "mtype": "numpy1D",
            "python_dependencies": None,
            "lossy": None,
        }
    
        def build(self):
            return None


class MultivTableSeries(MultivTable):

    _tags = {
        "mtype": "pd_Series_Table",
        "python_dependencies": None,
        "lossy": None,
    }

    def build(self):
        return None


class MultivTableListOfDict(MultivTable):

    _tags = {
        "mtype": "list_of_dict",
        "python_dependencies": None,
        "lossy": False,
    }

    def build(self):
        return [
            {"a": 1.0, "b": 3.0},
            {"a": 4.0, "b": 7.0},
            {"a": 0.5, "b": 2.0},
            {"a": -3.0, "b": -3 / 7},
        ]


class MultivTablePolarsEager(MultivTable):

    _tags = {
        "mtype": "polars_eager_table",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from skpro.datatypes._adapter.polars import convert_pandas_to_polars_with_index

        df = pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})
        return convert_pandas_to_polars_with_index(df)


class MultivTablePolarsLazy(MultivTable):

    _tags = {
        "mtype": "polars_lazy_table",
        "python_dependencies": ["polars", "pyarrow"],
        "lossy": False,
    }

    def build(self):
        from skpro.datatypes._adapter.polars import convert_pandas_to_polars_with_index

        df = pd.DataFrame({"a": [1, 4, 0.5, -3], "b": [3, 7, 2, -3 / 7]})
        return convert_pandas_to_polars_with_index(df, lazy=True)
