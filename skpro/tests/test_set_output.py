import pytest

from skpro.datatypes._table._convert import convert_pandas_to_polars_eager

# from skpro.tests.test_switch import run_test_module_changed
# from skpro.utils.set_output import check_output_config  # SUPPORTED_OUTPUTS,
from skpro.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    # import polars as pl
    pass

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split


@pytest.fixture
def polars_load_diabetes_pandas():
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
def polars_load_diabetes_polars(polars_load_diabetes_pandas):
    X_train, X_test, y_train = polars_load_diabetes_pandas
    X_train_pl = convert_pandas_to_polars_eager(X_train)
    X_test_pl = convert_pandas_to_polars_eager(X_test)
    y_train_pl = convert_pandas_to_polars_eager(y_train)

    # drop the index in the polars frame
    X_train_pl = X_train_pl.drop(["__index__"])
    X_test_pl = X_test_pl.drop(["__index__"])
    y_train_pl = y_train_pl.drop(["__index__"])

    return [X_train_pl, X_test_pl, y_train_pl]


# @pytest.fixture
# def estimator():
#     from sklearn.linear_model import LinearRegression

#     from skpro.regression.residual import ResidualDouble

#     # refactor to use ResidualDouble with Linear Regression
#     _estimator = ResidualDouble(LinearRegression())
#     return _estimator


# def test_check_transform_config_happy(estimator):
#     # check to make sure that regression estimators have the transform config
#     # with default value "default"
#     assert estimator.get_config()["transform"] == "default"

#     estimator.set_output(transform="pandas")
#     assert estimator.get_config()["transform"] == "pandas"
#     valid, dense_config = check_output_config(estimator)
#     assert valid
#     assert dense_config["dense"] == ("pd_DataFrame_Table", "Table")

#     if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
#         estimator.set_output(transform="polars")
#         assert estimator.get_config()["transform"] == "polars"
#         valid, dense_config = check_output_config(estimator)
#         assert valid
#         assert dense_config["dense"] == ("polars_eager_table", "Table")


# def test_check_transform_config_negative(estimator):
#     estimator.set_output(transform="foo")
#     with pytest.raises(
#         ValueError,
#         # match=f"set_output container must be in {SUPPORTED_OUTPUTS}, found foo.",
#     ):
#         check_output_config(estimator)


# def test_check_transform_config_none(estimator):
#     valid, dense = check_output_config(estimator)
#     assert not valid
#     assert dense == {}


# @pytest.mark.skipif(
#     not run_test_module_changed("skpro.datatypes")
#     or not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
#     reason="skip test if polars/pyarrow is not installed in environment",
# )
# def test_set_output_pandas_polars(polars_load_diabetes_pandas, estimator):
#     X_train, X_test, y_train = polars_load_diabetes_pandas
#     estimator.fit(X_train, y_train)
#     estimator.set_output(transform="polars")

#     y_pred = estimator.predict(X_test)
#     assert isinstance(y_pred, pl.DataFrame)

#     y_pred_interval = estimator.predict_interval(X_test)
#     assert isinstance(y_pred_interval, pl.DataFrame)

#     y_pred_quantiles = estimator.predict_quantiles(X_test)
#     assert isinstance(y_pred_quantiles, pl.DataFrame)


# @pytest.mark.skipif(
#     not run_test_module_changed("skpro.datatypes")
#     or not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
#     reason="skip test if polars/pyarrow is not installed in environment",
# )
# def test_set_output_polars_pandas(polars_load_diabetes_polars, estimator):
#     X_train_pl, X_test_pl, y_train_pl = polars_load_diabetes_polars
#     estimator.fit(X_train_pl, y_train_pl)
#     estimator.set_output(transform="pandas")

#     y_pred = estimator.predict(X_test_pl)
#     assert isinstance(y_pred, pd.DataFrame)

#     y_pred_interval = estimator.predict_interval(X_test_pl)
#     assert isinstance(y_pred_interval, pd.DataFrame)

#     y_pred_quantiles = estimator.predict_quantiles(X_test_pl)
#     assert isinstance(y_pred_quantiles, pd.DataFrame)
