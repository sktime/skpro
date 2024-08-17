"""Utility file for polars dataframes."""
from skpro.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    import polars as pl

    def polars_split_index_values_frame(obj):
        """Split all index and value columns into separate polars frame.

        Assumes there exists columns of the form __index__ inside the dataframe and
        other columns not in the double underscore form
        For example:

        ┌───────────┬────────────┐
        │ __index__ ┆ target     │
        │ ---       ┆ ---        │
        │ i64       ┆ f64        │
        ╞═══════════╪════════════╡
        │ 4         ┆ 121.545815 │
        │ 63        ┆ 77.2909    │
        │ 10        ┆ 74.845273  │
        └───────────┴────────────┘

        This function will then split the dataframes into 2 different
        polars dataframes

        ┌───────────┬     ┬────────────┐
        │ __index__ ┆     ┆ target     │
        │ ---       ┆     ┆ ---        │
        │ i64       ┆     ┆ f64        │
        ╞═══════════╪     ╪════════════╡
        │ 4         ┆     ┆ 121.545815 │
        │ 63        ┆     ┆ 77.2909    │
        │ 10        ┆     ┆ 74.845273  │
        └───────────┴     ┴────────────┘

        Parameters
        ----------
        obj: polars DataFrame
            has an assumption of the format of the dataframe


        Returns
        -------
        polars_index_frame: polars DataFrame
            polars frame containing only the index columns

        polars_values_frame: polars DataFrame
            polars frame containing only the value columns
        """
        obj_columns = obj.columns

        index_cols = [col for col in obj_columns if "__index__" in col]
        value_cols = [col for col in obj_columns if "__index__" not in col]

        polars_index_frame = obj.select(index_cols)
        polars_value_frame = obj.select(value_cols)

        return polars_index_frame, polars_value_frame

    def polars_combine_index_value_frame(polars_index_frame, polars_values_frame):
        """Combine the index and value frame together into a single frame.

        Parameter
        ---------
        polars_index_frame: polars DataFrame
            polars frame containing only the index columns

        polars_values_frame: polars DataFrame
            polars frame containing only the value columns

        Returns
        -------
        obj: polars DataFrame
            polars DataFrame containing both the index and value frames together
        """
        obj = pl.concat([polars_index_frame, polars_values_frame], how="horizontal")
        return obj
