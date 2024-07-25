"""Test file for polars dataframes"""

import pandas as pd
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from skpro.utils.validation._dependencies import _check_soft_dependencies

if _check_soft_dependencies(["polars", "pyarrow"], severity="none"):
    import polars as pl

    from skpro.datatypes._table._check import check_polars_table
    from skpro.datatypes._table._convert import convert_pandas_to_polars_eager

TEST_ALPHAS = [0.05, 0.1, 0.25]


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
def estimator():
    from sklearn.linear_model import LinearRegression

    from skpro.regression.residual import ResidualDouble

    # refactor to use ResidualDouble with Linear Regression
    _estimator = ResidualDouble(LinearRegression())
    return _estimator


@pytest.fixture
def polars_load_diabetes_polars(polars_load_diabetes_pandas):
    X_train, X_test, y_train = polars_load_diabetes_pandas
    X_train_pl = convert_pandas_to_polars_eager(X_train)
    X_test_pl = convert_pandas_to_polars_eager(X_test)
    y_train_pl = convert_pandas_to_polars_eager(y_train)

    return [X_train_pl, X_test_pl, y_train_pl]


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_polars_eager_conversion_methods(
    polars_load_diabetes_pandas, polars_load_diabetes_polars
):
    """
    Tests to ensure that given a pandas dataframe, the conversion methods
    convert properly to polars dataframe
    """

    X_train, X_test, y_train = polars_load_diabetes_pandas
    X_train_pl, X_test_pl, y_train_pl = polars_load_diabetes_polars

    assert check_polars_table(X_train_pl)
    assert check_polars_table(X_test_pl)
    assert check_polars_table(y_train_pl)
    assert (X_train.values == X_train_pl.to_numpy()).all()
    assert (X_test.values == X_test_pl.to_numpy()).all()
    assert (y_train.values == y_train_pl.to_numpy()).all()


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_polars_eager_regressor_in_fit_predict(
    estimator, polars_load_diabetes_pandas, polars_load_diabetes_polars
):
    """
    Tests to ensure that given a polars dataframe, the regression estimator
    can fit and predict and return the correct set of outputs

    Parameters
    ----------

    estimator: a given regression estimator

    """
    # TODO - expand estimator to include a list of regression models to test
    # create a copy of estimator to run further checks
    estimator_copy = estimator
    X_train, X_test, y_train = polars_load_diabetes_pandas
    X_train_pl, X_test_pl, y_train_pl = polars_load_diabetes_polars

    estimator.fit(X_train_pl, y_train_pl)
    y_pred = estimator.predict(X_test_pl)

    assert isinstance(y_pred, pl.DataFrame)
    assert y_pred.columns == y_train_pl.columns
    assert y_pred.shape[0] == X_test_pl.shape[0]

    # code to ensure prediction values match up correctly
    estimator_copy.fit(X_train, y_train)
    y_pred_pd = estimator_copy.predict(X_test)
    assert (y_pred_pd.values == y_pred.to_numpy()).all()


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_polars_eager_regressor_in_predict_interval(
    estimator, polars_load_diabetes_polars
):
    X_train_pl, X_test_pl, y_train_pl = polars_load_diabetes_polars
    # TODO - expand estimator to include a list of regression models to test
    estimator.fit(X_train_pl, y_train_pl)
    y_pred_interval = estimator.predict_interval(X_test_pl)

    assert isinstance(y_pred_interval, pd.DataFrame)
    assert y_pred_interval.columns[0] == ("target", 0.9, "lower")
    assert y_pred_interval.columns[1] == ("target", 0.9, "upper")


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_polars_eager_regressor_in_predict_quantiles(
    estimator, polars_load_diabetes_polars
):
    X_train_pl, X_test_pl, y_train_pl = polars_load_diabetes_polars

    estimator.fit(X_train_pl, y_train_pl)
    y_pred_quantile = estimator.predict_quantiles(X_test_pl, alpha=TEST_ALPHAS)

    assert isinstance(y_pred_quantile, pd.DataFrame)
    assert y_pred_quantile.columns[0] == ("target", 0.05)
    assert y_pred_quantile.columns[1] == ("target", 0.1)
    assert y_pred_quantile.columns[2] == ("target", 0.25)
