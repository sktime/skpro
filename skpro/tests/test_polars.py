"""Test file for polars dataframes"""

import pytest

# from skpro.datatypes._table._convert import (
#     convert_polars_to_pandas,
#     convert_pandas_to_polars_eager
# )
from sktime.utils.validation._dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("polars", severity="none"),
    reason="skip test if polars is not installed in environment",
)
def test_polars_dataframe_in_fit():
    pass


def test_polars_dataframe_in_predict():
    pass
