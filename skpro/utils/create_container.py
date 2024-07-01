"""Methods for create_container functionality."""
# this is a temp file to store potential conversion methods and will be deleted once
# the data types module in 392 is implemented

# for now, will call it containerAdapter, but will probably be merged
# along with 392
import pandas as pd

from skpro.utils.set_output import (
    check_n_level_of_dataframe,
    convert_pandas_multiindex_columns_to_single_column,
)
from skpro.utils.validation._dependencies import _check_soft_dependencies


def get_config_adapter(transform_adapter, X):
    """Return a given adapter and columns."""
    if transform_adapter == "pandas":
        adapter = PandasAdapter
    if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
        if transform_adapter == "polars":
            adapter = PolarsAdapter

    # need to convert the columns beforehand
    if hasattr(X, "columns"):
        levels = check_n_level_of_dataframe(X)
        # if levels > 1, must be Pandas DataFrames
        if levels > 1:
            columns = convert_pandas_multiindex_columns_to_single_column(X)
        elif levels == 1:
            columns = X.columns
    else:
        columns = None

    return adapter, columns


def rename_columns(X, columns):
    """Rename a given set of columns in a data container."""
    X.columns = columns


class PandasAdapter:
    """Pandas Adapter."""

    container_name = "pandas"

    def create_container(X_input, columns):
        """Create output data container in Pandas format."""
        out = pd.DataFrame(X_input)
        if columns is not None:
            rename_columns(out, columns)

        return out


if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    import polars as pl

    class PolarsAdapter:
        """Polars Adapter."""

        container_name = "polars"

        def create_container(X_input, columns):
            """Create output data container in Polars format."""
            out = pl.DataFrame(X_input)
            if columns is not None:
                rename_columns(out, columns)

            return out
