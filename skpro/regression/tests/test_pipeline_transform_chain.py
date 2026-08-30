import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer

from skpro.regression.compose import Pipeline
from skpro.regression.residual import ResidualDouble


def test_transformer_chaining_in_predict():
    """Ensure transformers are applied sequentially in pipeline."""

    X = pd.DataFrame({"x": np.arange(5)})
    y = pd.Series(np.arange(5))

    exp = FunctionTransformer(np.exp)

    pipe = Pipeline(
        [
            ("exp1", exp),
            ("exp2", exp),
            ("reg", ResidualDouble(LinearRegression())),
        ]
    )

    pipe.fit(X, y)

    # run predict to ensure the pipeline works end-to-end
    y_pred = pipe.predict(X)

    # check that transformations were applied sequentially
    Xt = pipe._transform(X)
    expected = np.exp(np.exp(X))

    assert np.allclose(Xt.values, expected.values)
    assert len(y_pred) == len(y)


def test_repeated_predict_numpy_X_named_y_series():
    """Regression test: second predict must not crash due to _X_converter_store
    cross-contamination when y is a named pd.Series and X is 1-feature numpy.

    Before the fix, the y output converter in predict() incorrectly wrote y column
    names into _X_converter_store. On the second call, _check_X() read the
    contaminated store and gave X the wrong column name, crashing the inner regressor.
    """
    X_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y_train = pd.Series([1.1, 2.0, 3.1, 3.9, 5.0], name="target")

    reg = ResidualDouble(LinearRegression())
    reg.fit(X_train, y_train)

    X_test = np.array([[6.0], [7.0]])

    y_pred_1 = reg.predict(X_test)
    y_pred_2 = reg.predict(X_test)

    assert y_pred_1.values.tolist() == y_pred_2.values.tolist()
    assert len(y_pred_2) == len(X_test)
