"""Regression tests for bugs in the regressor base class."""

import numpy as np
import pandas as pd
import pytest

from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.regression"),
    reason="run only if skpro.regression has been changed",
)
def test_predict_quantiles_from_interval_multivariate():
    """Regression test for quantiles defaulting to intervals, multivariate case.

    Quantiles derived from ``_predict_interval`` were assigned to the wrong
    variables and alphas if ``y`` had more than one variable.
    """
    from skpro.regression.base import BaseProbaRegressor

    class _IntervalOnlyRegressor(BaseProbaRegressor):
        """Regressor whose interval for variable j has half-width 10 * (j+1) * c."""

        _tags = {"capability:multioutput": True}

        def _fit(self, X, y):
            self.means_ = y.mean().values
            return self

        def _predict(self, X):
            values = np.tile(self.means_, (len(X), 1))
            return pd.DataFrame(
                values, index=X.index, columns=self._y_metadata["feature_names"]
            )

        def _predict_interval(self, X, coverage):
            columns = self._get_columns(method="predict_interval", coverage=coverage)
            values = []
            for j, mean in enumerate(self.means_):
                for c in coverage:
                    values += [mean - 10 * (j + 1) * c, mean + 10 * (j + 1) * c]
            values = np.tile(values, (len(X), 1))
            return pd.DataFrame(values, index=X.index, columns=columns)

    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [100.0, 101.0, 102.0], "c": [-5.0, -5.0, -5.0]}
    )
    alpha = [0.05, 0.25, 0.75, 0.95]

    reg = _IntervalOnlyRegressor().fit(X, y)
    pred_q = reg.predict_quantiles(X, alpha=alpha)

    assert list(pred_q.columns) == [(var, a) for var in y.columns for a in alpha]
    for j, var in enumerate(y.columns):
        mean = y[var].mean()
        for a in alpha:
            expected = mean + 10 * (j + 1) * (2 * a - 1)
            np.testing.assert_allclose(pred_q[(var, a)].values, expected)
