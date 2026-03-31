"""Registry of mtypes for Table scitype.

See datatypes._registry for API.
"""

import pandas as pd

__all__ = [
    "MTYPE_REGISTER_TABLE",
    "MTYPE_LIST_TABLE",
]


MTYPE_REGISTER_TABLE = [
    (
        "pd_DataFrame_Table",
        "Table",
        "pd.DataFrame representation of a data table",
        None,
    ),
    ("numpy1D", "Table", "1D np.ndarray representation of a univariate table", None),
    ("numpy2D", "Table", "2D np.ndarray representation of a univariate table", None),
    ("pd_Series_Table", "Table", "pd.Series representation of a data table", None),
    ("list_of_dict", "Table", "list of dictionaries with primitive entries", None),
    (
        "polars_eager_table",
        "Table",
        "polars.DataFrame representation of a data table",
        ["polars", "pyarrow"],
    ),
    (
        "polars_lazy_table",
        "Table",
        "polars.LazyFrame representation of a data table",
        ["polars", "pyarrow"],
    ),
]

MTYPE_LIST_TABLE = pd.DataFrame(MTYPE_REGISTER_TABLE)[0].values
