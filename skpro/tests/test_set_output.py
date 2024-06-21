import pandas as pd
import pytest

from skpro.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    import polars as pl

from skpro.utils.set_output import (
    check_column_level_of_dataframe,
    check_transform_config,
    convert_multiindex_columns_to_single_column,
    convert_pandas_dataframe_to_polars_eager_with_index,
    convert_pandas_index_to_column,
)


@pytest.fixture
def load_pandas_multi_index_column_fixture():
    arrays = [
        ["A", "A", "A", "A"],
        ["Foo", "Foo", "Bar", "Bar"],
        ["One", "Two", "One", "Two"],
    ]
    columns = pd.MultiIndex.from_arrays(arrays)

    # Create the DataFrame
    data = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    pd_multi_column_fixture = pd.DataFrame(data, columns=columns)

    return pd_multi_column_fixture


@pytest.fixture
def load_pandas_simple_column_fixture():
    data = {"test_target": [10, 20, 30]}

    # Create the DataFrame with a custom index
    pd_simple_column_fixture = pd.DataFrame(
        data, index=pd.Index(["row1", "row2", "row3"], name="foo")
    )

    return pd_simple_column_fixture


@pytest.fixture
def load_polars_simple_fixture():
    data = {"column1": [1, 2, 3], "column2": [4, 5, 6], "column3": [7, 8, 9]}

    # Create the DataFrame
    pl_simple_fixture = pl.DataFrame(data)

    return pl_simple_fixture


@pytest.fixture
def estimator():
    from sklearn.linear_model import LinearRegression

    from skpro.regression.residual import ResidualDouble

    # refactor to use ResidualDouble with Linear Regression
    _estimator = ResidualDouble(LinearRegression())
    return _estimator


def test_check_column_level_of_dataframe(
    load_pandas_multi_index_column_fixture,
    load_pandas_simple_column_fixture,
    load_polars_simple_fixture,
):
    pd_multi_column_fixture = load_pandas_multi_index_column_fixture()
    pd_simple_column_fixture = load_pandas_simple_column_fixture()

    n_levels_multi_pd = check_column_level_of_dataframe(pd_multi_column_fixture)
    n_levels_simple_pd = check_column_level_of_dataframe(pd_simple_column_fixture)

    assert n_levels_multi_pd == 3
    assert n_levels_simple_pd == 1

    if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
        pl_simple_column_fixture = load_polars_simple_fixture()
        n_levels_simple_pl = check_column_level_of_dataframe(pl_simple_column_fixture)

        assert n_levels_simple_pl == 1


def test_convert_multiindex_columns_to_single_column(
    load_pandas_multi_index_column_fixture,
):
    pd_multi_column_fixture = load_pandas_multi_index_column_fixture
    df_list = convert_multiindex_columns_to_single_column(pd_multi_column_fixture)
    assert df_list == [
        "__A__Foo__One__",
        "__A__Foo__Two__",
        "__A__Bar__One__",
        "__A__Bar__Two__",
    ]


def test_check_transform_config_happy(estimator):
    # check to make sure that regression estimators have the transform config
    # with default value None
    assert not estimator.get_config()["transform"]

    estimator.set_output(transform="pandas")
    assert estimator.get_config()["transform"] == "pandas"
    assert check_transform_config(estimator)["dense"] == "pandas"

    estimator.set_output(transform="polars")
    assert estimator.get_config()["transform"] == "polars"
    assert check_transform_config(estimator)["dense"] == "polars"


def test_check_transform_config_negative():
    estimator.set_output(transform="foo")
    with pytest.raises(
        ValueError,
        match='set_output container must be in ["polars", "pandas"], found foo.',
    ):
        check_transform_config(estimator)


def test_convert_pandas_dataframe_to_polars_eager_with_index():
    convert_pandas_dataframe_to_polars_eager_with_index


def test_convert_pandas_index_to_column():
    convert_pandas_index_to_column
