"""Test file for polars dataframes"""

import pytest
from sktime.utils.validation._dependencies import _check_soft_dependencies

from skpro.datatypes._table._convert import convert_pandas_to_polars_eager


@pytest.mark.skipif(
    not _check_soft_dependencies(["polars", "pyarrow"], severity="none"),
    reason="skip test if polars/pyarrow is not installed in environment",
)
def test_polars_dataframe_in_fit(estimator):
    import pandas as pd
    import polars as pl
    from sklearn.datasets import load_diabetes
    from sklearn.model_selection import train_test_split

    X, y = load_diabetes(return_X_y=True, as_frame=True)
    X = X.iloc[:75]
    y = y.iloc[:75]
    y = pd.DataFrame(y)

    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    X_train_pl = convert_pandas_to_polars_eager(X_train)
    X_test_pl = convert_pandas_to_polars_eager(X_test)
    y_train_pl = convert_pandas_to_polars_eager(y_train)

    estimator.fit(X_train_pl, y_train_pl)

    # test predict output contract
    y_pred = estimator.predict(X_test_pl)

    assert isinstance(y_pred, pl.DataFrame)
    assert (y_pred.columns == y_train_pl.columns).all()
