import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer

from skpro.regression.compose import Pipeline
from skpro.regression.residual import ResidualDouble
from skpro.tests.test_switch import run_test_for_class


@pytest.mark.skipif(
    not run_test_for_class([Pipeline, ResidualDouble]),
    reason="run test only if tested object has changed",
)
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
