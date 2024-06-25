import pandas as pd
import pytest

from skpro.utils.validation._dependencies import _check_soft_dependencies

# TODO - write functions that ensures that the column values from the single level
# columsn frame matches the multi-index columsn frame


if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    import polars as pl

from skpro.utils.set_output import (  # SUPPORTED_OUTPUTS,
    check_n_level_of_dataframe,
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
        data, index=pd.Index(["row1", "row2", "row3"])
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


def test_check_column_level_of_dataframe_pandas(
    load_pandas_multi_index_column_fixture,
    load_pandas_simple_column_fixture,
):
    pd_multi_column_fixture = load_pandas_multi_index_column_fixture
    pd_simple_column_fixture = load_pandas_simple_column_fixture

    n_levels_multi_pd = check_n_level_of_dataframe(pd_multi_column_fixture)
    n_levels_simple_pd = check_n_level_of_dataframe(pd_simple_column_fixture)

    assert n_levels_multi_pd == 3
    assert n_levels_simple_pd == 1


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_check_column_level_of_dataframe_polars(
    load_polars_simple_fixture,
):
    pl_simple_column_fixture = load_polars_simple_fixture
    n_levels_simple_pl = check_n_level_of_dataframe(pl_simple_column_fixture)
    assert n_levels_simple_pl == 1


def test_convert_multiindex_columns_to_single_column(
    load_pandas_multi_index_column_fixture,
):
    pd_multi_column_fixture1 = load_pandas_multi_index_column_fixture
    df_list1 = convert_multiindex_columns_to_single_column(pd_multi_column_fixture1)
    assert df_list1 == [
        "__A__Foo__One__",
        "__A__Foo__Two__",
        "__A__Bar__One__",
        "__A__Bar__Two__",
    ]

    pd_multi_column_fixture2 = load_pandas_multi_index_column_fixture
    pd_multi_column_fixture2 = convert_pandas_index_to_column(pd_multi_column_fixture2)
    df_list2 = convert_multiindex_columns_to_single_column(pd_multi_column_fixture2)
    assert df_list2 == [
        "__index__",
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
    valid, dense_config = check_transform_config(estimator)
    assert valid
    assert dense_config["dense"] == "pandas"

    if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
        estimator.set_output(transform="polars")
        assert estimator.get_config()["transform"] == "polars"
        valid, dense_config = check_transform_config(estimator)
        assert valid
        assert dense_config["dense"] == "polars"


def test_check_transform_config_negative(estimator):
    estimator.set_output(transform="foo")
    with pytest.raises(
        ValueError,
        # match=f"set_output container must be in {SUPPORTED_OUTPUTS}, found foo.",
    ):
        check_transform_config(estimator)


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_convert_pandas_dataframe_to_polars_eager_with_index(
    load_pandas_simple_column_fixture,
):
    pd_simple_column_fixture1 = load_pandas_simple_column_fixture
    pd_simple_column_fixture2 = pd_simple_column_fixture1.copy(deep=True)
    pd_simple_column_fixture2.index.name = "foo"

    X_out1 = convert_pandas_dataframe_to_polars_eager_with_index(
        pd_simple_column_fixture1, include_index=True
    )
    X_out2 = convert_pandas_dataframe_to_polars_eager_with_index(
        pd_simple_column_fixture2, include_index=True
    )
    X_out3 = convert_pandas_dataframe_to_polars_eager_with_index(
        pd_simple_column_fixture1, include_index=False
    )

    assert "__index__" in X_out1.columns
    assert "foo" in X_out2.columns

    assert all(X_out1["__index__"].to_numpy() == pd_simple_column_fixture1.index)
    assert all(X_out2["foo"].to_numpy() == pd_simple_column_fixture2.index)
    assert all(X_out3.to_numpy() == pd_simple_column_fixture1.values)


def test_convert_pandas_index_to_column(load_pandas_simple_column_fixture):
    pd_simple_column_fixture = load_pandas_simple_column_fixture
    X_out = convert_pandas_index_to_column(pd_simple_column_fixture)

    assert "__index__" in X_out.columns
    assert all(X_out["__index__"].values == pd_simple_column_fixture.index)
