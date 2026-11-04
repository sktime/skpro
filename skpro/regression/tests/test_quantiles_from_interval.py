# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Test default predict_quantiles via predict_interval, multioutput case."""

__author__ = ["VividhDesign"]

import numpy as np
import pandas as pd
import pytest

from skpro.regression.base import BaseProbaRegressor
from skpro.tests.test_switch import run_test_module_changed


class _IntervalOnlyRegressor(BaseProbaRegressor):
    """Regressor whose interval for variable j has half-width 10 * (j+1) * c."""

    _tags = {"capability:multioutput": True}

    def _fit(self, X, y):
        self.means_ = y.mean().values
        return self

    def _predict(self, X):
        values = np.tile(self.means_, (len(X), 1))
        return pd.DataFrame(values, index=X.index, columns=self._get_varnames())

    def _predict_interval(self, X, coverage):
        columns = self._get_columns(method="predict_interval", coverage=coverage)
        values = []
        for j, mean in enumerate(self.means_):
            for c in coverage:
                values += [mean - 10 * (j + 1) * c, mean + 10 * (j + 1) * c]
        values = np.tile(values, (len(X), 1))
        return pd.DataFrame(values, index=X.index, columns=columns)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.regression"),
    reason="run only if skpro.regression has been changed",
)
def test_predict_quantiles_from_interval_multioutput():
    """Quantiles defaulting to intervals must be assigned to the right variable."""
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0]})
    y = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [100.0, 101.0, 102.0], "c": [-5.0, -5.0, -5.0]}
    )
    alpha = [0.05, 0.25, 0.75, 0.95]

    reg = _IntervalOnlyRegressor().fit(X, y)
    pred_q = reg.predict_quantiles(X, alpha=alpha)

    for j, var in enumerate(y.columns):
        for a in alpha:
            # lower end for alpha < 0.5, upper end for alpha >= 0.5
            expected = y[var].mean() + 10 * (j + 1) * (2 * a - 1)
            np.testing.assert_allclose(pred_q[(var, a)].values, expected)
