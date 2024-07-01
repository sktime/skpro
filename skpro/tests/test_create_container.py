# test file for create_container methods
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from skpro.utils.create_container import (
    PandasAdapter,
    PolarsAdapter,
    get_config_adapter,
)
from skpro.utils.set_output import check_transform_config
from skpro.utils.validation._dependencies import _check_soft_dependencies


# test following cases, numpy input, polars input, pandas input
@pytest.fixture
def load_diabetes_pandas():
    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.iloc[:15]
    y = y.iloc[:15]

    # typically y is returned as a pd.Series to we call y as a Dataframe here
    y = pd.DataFrame(y)

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    return [X_train, X_test, y_train]


@pytest.fixture
def load_pandas_pred_quantile():
    data = np.array(
        [
            [4, 66.772658, 176.318973],
            [63, 22.517743, 132.064058],
            [10, 20.072116, 129.618431],
            [0, 177.079295, 286.625610],
            [35, 86.009941, 195.556256],
        ]
    )

    quantiles = pd.DataFrame(data[:, 1:], index=data[:, 0].astype(int))
    quantiles.columns = pd.MultiIndex.from_tuples([("target", 0.05), ("target", 0.95)])

    return quantiles


@pytest.fixture
def load_pandas_pred_interval():
    data = np.array(
        [
            [4, 66.772658, 176.318973],
            [63, 22.517743, 132.064058],
            [10, 20.072116, 129.618431],
            [0, 177.079295, 286.625610],
            [35, 86.009941, 195.556256],
        ]
    )

    pred_int = pd.DataFrame(data[:, 1:], index=data[:, 0].astype(int))
    pred_int.columns = pd.MultiIndex.from_tuples(
        [("target", 0.9, "upper"), ("target", 0.9, "lower")]
    )

    return pred_int


@pytest.fixture
def load_pandas_pred_var():
    data = {"target": [1108.871039, 1108.871039, 1108.871039, 1108.871039, 1108.871039]}

    pred_var = pd.DataFrame(data)
    pred_var.index = [4, 63, 10, 0, 35]

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


def test_create_container_to_pandas(estimator, load_diabetes_pandas):
    estimator.set_output(transform="pandas")
    X_train, X_test, y_train = load_diabetes_pandas

    estimator.fit(X_train, y_train)
    y_pred_q = estimator.predict_quantiles(X_test)[:5]
    y_pred_i = estimator.predict_interval(X_test)[:5]
    y_pred_v = estimator.predict_var(X_test)[:5]

    _, output_config = check_transform_config(estimator)
    transform_adapter = output_config["dense"]

    adapter, columns = get_config_adapter(transform_adapter, y_pred_q)
    y_pred_q_out = adapter.create_container(y_pred_q, columns)
    assert (y_pred_q_out.values == y_pred_q.values).all()
    assert (y_pred_q_out.columns == y_pred_q.columns).all()
    assert (y_pred_q_out.index == y_pred_q.index).all()

    adapter, columns = get_config_adapter(transform_adapter, y_pred_i)
    y_pred_i_out = adapter.create_container(y_pred_q, columns)
    assert (y_pred_i_out.values == y_pred_i.values).all()
    assert (y_pred_i_out.columns == y_pred_i.columns).all()
    assert (y_pred_i_out.index == y_pred_i.index).all()

    adapter, columns = get_config_adapter(transform_adapter, y_pred_v)
    y_pred_v_out = adapter.create_container(y_pred_v, columns)
    assert (y_pred_v_out.values == y_pred_v.values).all()
    assert (y_pred_v_out.columns == y_pred_v.columns).all()
    assert (y_pred_v_out.index == y_pred_v.index).all()


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_create_container_to_polars(estimator, load_diabetes_pandas):
    estimator.set_output(transform="polars")
    X_train, X_test, y_train = load_diabetes_pandas

    estimator.fit(X_train, y_train)
    y_pred_q = estimator.predict_quantiles(X_test)[:5]
    y_pred_i = estimator.predict_interval(X_test)[:5]
    y_pred_v = estimator.predict_var(X_test)[:5]

    _, output_config = check_transform_config(estimator)
    transform_adapter = output_config["dense"]

    adapter, columns = get_config_adapter(transform_adapter, y_pred_q)
    y_pred_q_out = adapter.create_container(y_pred_q, columns)
    assert (y_pred_q_out.to_numpy() == y_pred_q.values).all()
    assert y_pred_q_out.columns == ["__target__0.05__", "__target__0.95__"]

    adapter, columns = get_config_adapter(transform_adapter, y_pred_i)
    y_pred_i_out = adapter.create_container(y_pred_q, columns)
    assert (y_pred_i_out.to_numpy() == y_pred_i.values).all()
    assert y_pred_i_out.columns == ["__target__0.9__lower__", "__target__0.9__upper__"]

    adapter, columns = get_config_adapter(transform_adapter, y_pred_v)
    y_pred_v_out = adapter.create_container(y_pred_v, columns)
    assert (y_pred_v_out.to_numpy() == y_pred_v.values).all()
    assert y_pred_v_out.columns == ["target"]
