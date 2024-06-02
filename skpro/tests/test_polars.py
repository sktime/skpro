"""Test file for polars dataframes"""

import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from skpro.datatypes._table._convert import convert_pandas_to_polars_eager
from skpro.utils.validation._dependencies import _check_soft_dependencies

X, y = load_diabetes(return_X_y=True, as_frame=True)
X = X.iloc[:75]
y = y.iloc[:75]

# typically y is returned as a pd.Series to we call y as a Dataframe here
y = pd.DataFrame(y)

X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.33, random_state=42)


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_polars_eager_conversion_methods():
    """
    Tests to ensure that given a pandas dataframe, the conversion methods
    convert properly
    """
    # y_train returns a pandas series so the skpro conversion function does not work
    X_train_pl = convert_pandas_to_polars_eager(X_train)
    X_test_pl = convert_pandas_to_polars_eager(X_test)
    y_train_pl = convert_pandas_to_polars_eager(y_train)

    assert (X_train.values == X_train_pl.to_numpy()).all()
    assert (X_test.values == X_test_pl.to_numpy()).all()
    assert (y_train.values == y_train_pl.to_numpy()).all()


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_polars_eager_regressor_in_fit_predict(estimator):
    """
    Tests to ensure that given a polars dataframe, the regression estimator
    can fit and predict and return the correct set of outputs

    Parameters
    ----------

    estimator: a given regression estimator

    """
    # create a copy of estimator to run further checks
    estimator_copy = estimator

    X_train_pl = convert_pandas_to_polars_eager(X_train)
    X_test_pl = convert_pandas_to_polars_eager(X_test)
    y_train_pl = convert_pandas_to_polars_eager(y_train)

    estimator.fit(X_train_pl, y_train_pl)
    y_pred = estimator.predict(X_test_pl)

    assert isinstance(y_pred, pl.DataFrame)
    assert y_pred.columns == y_train_pl.columns
    assert y_pred.shape[0] == X_test_pl.shape[0]

    # code to ensure prediction values match up correctly
    estimator_copy.fit(X_train, y_train)
    y_pred_pd = estimator_copy.predict(X_test)
    assert (y_pred_pd.values == y_pred.to_numpy()).all()
