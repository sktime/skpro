# test file for create_container methods
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from skpro.utils.create_container import PandasAdapter, get_config_adapter
from skpro.utils.set_output import check_transform_config
from skpro.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    import polars as pl

    from skpro.utils.create_container import PolarsAdapter


# test following cases, numpy input, polars input, pandas input
@pytest.fixture
def load_diabetes_pandas():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.iloc[:75]
    y = y.iloc[:75]

    # typically y is returned as a pd.Series to we call y as a Dataframe here
    y = pd.DataFrame(y)

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return [X_train, X_test, y_train]


@pytest.fixture
def load_pandas_pred():
    data = np.array(
        [
            [4, 121.545815],
            [63, 77.290900],
            [10, 74.845273],
            [0, 231.852453],
            [35, 140.783099],
        ]
    )
    data = np.round(data, 2)
    pred = pd.DataFrame(data[:, 1:], index=data[:, 0].astype(int))
    pred.columns = ["target"]

    return pred


@pytest.fixture
def load_pandas_pred_quantile():
    data = np.array(
        [
            [4, 66.77265782260743, 176.3189727940714],
            [63, 22.51774261798211, 132.0640575894461],
            [10, 20.072115541571705, 129.6184305130357],
            [0, 177.07929524984633, 286.6256102213103],
            [35, 86.00994115791546, 195.55625612937945],
        ]
    )
    data = np.round(data, 2)
    quantiles = pd.DataFrame(data[:, 1:], index=data[:, 0].astype(int))
    quantiles.columns = pd.MultiIndex.from_tuples([("target", 0.05), ("target", 0.95)])

    return quantiles


@pytest.fixture
def load_pandas_pred_interval():
    data = np.array(
        [
            [4, 66.77265782260743, 176.3189727940714],
            [63, 22.51774261798211, 132.0640575894461],
            [10, 20.072115541571705, 129.6184305130357],
            [0, 177.07929524984633, 286.6256102213103],
            [35, 86.00994115791546, 195.55625612937945],
        ]
    )
    data = np.round(data, 2)
    pred_int = pd.DataFrame(data[:, 1:], index=data[:, 0].astype(int))
    pred_int.columns = pd.MultiIndex.from_tuples(
        [("target", 0.9, "lower"), ("target", 0.9, "upper")]
    )

    return pred_int


@pytest.fixture
def load_pandas_pred_var():
    data = np.array(
        [
            1108.8710389831333,
            1108.8710389831333,
            1108.8710389831333,
            1108.8710389831333,
            1108.8710389831333,
        ]
    )
    data = np.round(data, 2)
    pred_var = pd.DataFrame(data)
    pred_var.index = [4, 63, 10, 0, 35]
    pred_var.columns = ["target"]
    return pred_var


@pytest.fixture
def estimator():
    from sklearn.linear_model import LinearRegression

    from skpro.regression.residual import ResidualDouble

    # refactor to use ResidualDouble with Linear Regression
    _estimator = ResidualDouble(LinearRegression())
    return _estimator


def test_get_config_adapter_pandas(estimator, load_diabetes_pandas):
    estimator.set_output(transform="pandas")
    X_train, X_test, y_train = load_diabetes_pandas

    estimator.fit(X_train, y_train)
    y_pred_q = estimator.predict_quantiles(X_test)[:5]
    y_pred_i = estimator.predict_interval(X_test)[:5]
    y_pred_v = estimator.predict_var(X_test)[:5]

    _, output_config = check_transform_config(estimator)
    transform_adapter = output_config["dense"]

    adapter, columns = get_config_adapter(transform_adapter, y_pred_q)
    assert isinstance(adapter, PandasAdapter)
    assert (columns == y_pred_q.columns).all()

    adapter, columns = get_config_adapter(transform_adapter, y_pred_i)
    assert isinstance(adapter, PandasAdapter)
    assert (columns == y_pred_i.columns).all()

    adapter, columns = get_config_adapter(transform_adapter, y_pred_v)
    assert isinstance(adapter, PandasAdapter)
    assert (columns == y_pred_v.columns).all()


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_get_config_adapter_polars(estimator, load_diabetes_pandas):
    estimator.set_output(transform="polars")
    X_train, X_test, y_train = load_diabetes_pandas

    estimator.fit(X_train, y_train)
    y_pred_q = estimator.predict_quantiles(X_test)[:5]
    y_pred_i = estimator.predict_interval(X_test)[:5]
    y_pred_v = estimator.predict_var(X_test)[:5]

    _, output_config = check_transform_config(estimator)
    transform_adapter = output_config["dense"]

    adapter, columns = get_config_adapter(transform_adapter, y_pred_q)
    assert isinstance(adapter, PolarsAdapter)
    assert columns == ["__target__0.05__", "__target__0.95__"]

    adapter, columns = get_config_adapter(transform_adapter, y_pred_i)
    assert isinstance(adapter, PolarsAdapter)
    assert columns == ["__target__0.9__lower__", "__target__0.9__upper__"]

    adapter, columns = get_config_adapter(transform_adapter, y_pred_v)
    assert isinstance(adapter, PolarsAdapter)
    assert columns == ["target"]


def test_create_container_to_pandas(
    estimator,
    load_pandas_pred,
    load_diabetes_pandas,
    load_pandas_pred_quantile,
    load_pandas_pred_interval,
    load_pandas_pred_var,
):
    estimator.set_output(transform="pandas")
    expected_y_pred = load_pandas_pred
    expected_y_pred_int = load_pandas_pred_interval
    expected_y_pred_quantile = load_pandas_pred_quantile
    expected_y_pred_var = load_pandas_pred_var
    X_train, X_test, y_train = load_diabetes_pandas

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)[:5]
    y_pred_q = estimator.predict_quantiles(X_test)[:5]
    y_pred_i = estimator.predict_interval(X_test)[:5]
    y_pred_v = estimator.predict_var(X_test)[:5]

    assert isinstance(y_pred, pd.DataFrame)
    assert isinstance(y_pred_q, pd.DataFrame)
    assert isinstance(y_pred_i, pd.DataFrame)
    assert isinstance(y_pred_v, pd.DataFrame)

    assert (expected_y_pred.values == np.round(y_pred.values, 2)).all()
    assert (expected_y_pred.columns == y_pred.columns).all()
    assert (expected_y_pred.index == y_pred.index).all()

    assert (expected_y_pred_quantile.values == np.round(y_pred_q.values, 2)).all()
    assert (expected_y_pred_quantile.columns == y_pred_q.columns).all()
    assert (expected_y_pred_quantile.index == y_pred_q.index).all()

    assert (expected_y_pred_quantile.values == np.round(y_pred_q.values, 2)).all()
    assert (expected_y_pred_quantile.columns == y_pred_q.columns).all()
    assert (expected_y_pred_quantile.index == y_pred_q.index).all()

    assert (expected_y_pred_int.values == np.round(y_pred_i.values, 2)).all()
    assert (expected_y_pred_int.columns == y_pred_i.columns).all()
    assert (expected_y_pred_int.index == y_pred_i.index).all()

    assert (expected_y_pred_var.values == np.round(y_pred_v.values, 2)).all()
    assert (expected_y_pred_var.columns == y_pred_v.columns).all()
    assert (expected_y_pred_var.index == y_pred_v.index).all()


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_create_container_to_polars(
    estimator,
    load_pandas_pred,
    load_diabetes_pandas,
    load_pandas_pred_quantile,
    load_pandas_pred_interval,
    load_pandas_pred_var,
):
    estimator.set_output(transform="polars")
    expected_y_pred = load_pandas_pred
    expected_y_pred_int = load_pandas_pred_interval
    expected_y_pred_quantile = load_pandas_pred_quantile
    expected_y_pred_var = load_pandas_pred_var
    X_train, X_test, y_train = load_diabetes_pandas

    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)[:5]
    y_pred_q = estimator.predict_quantiles(X_test)[:5]
    y_pred_i = estimator.predict_interval(X_test)[:5]
    y_pred_v = estimator.predict_var(X_test)[:5]

    assert isinstance(y_pred, pl.DataFrame)
    assert isinstance(y_pred_q, pl.DataFrame)
    assert isinstance(y_pred_i, pl.DataFrame)
    assert isinstance(y_pred_v, pl.DataFrame)

    assert (np.round(y_pred.to_numpy(), 2) == expected_y_pred.values).all()
    assert y_pred.columns == ["target"]

    assert (np.round(y_pred_q.to_numpy(), 2) == expected_y_pred_quantile.values).all()
    assert y_pred_q.columns == ["__target__0.05__", "__target__0.95__"]

    assert (np.round(y_pred_i.to_numpy(), 2) == expected_y_pred_int.values).all()
    assert y_pred_i.columns == [
        "__target__0.9__lower__",
        "__target__0.9__upper__",
    ]

    assert (np.round(y_pred_v.to_numpy(), 2) == expected_y_pred_var.values).all()
    assert y_pred_v.columns == ["target"]
