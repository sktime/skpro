"""Tests for CoxPH statsmodels adapter."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

import pytest


@pytest.fixture
def coxph_strata_data():
    """Return minimal data with strata for CoxPH tests."""
    import pandas as pd

    X = pd.DataFrame({"x": [0.0, 1.0] * 6, "group": [0] * 6 + [1] * 6})
    y = pd.DataFrame(
        {
            "time": [
                1.0,
                3.0,
                4.0,
                2.0,
                5.0,
                6.0,
                2.0,
                4.0,
                3.0,
                6.0,
                5.0,
                7.0,
            ]
        }
    )
    C = pd.DataFrame({"time": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0]})
    return X, y, C


def test_coxph_strata_predict_proba(coxph_strata_data):
    """Regression test for GitHub issue #1216.

    CoxPH with strata parameter should be able to call predict_proba after fit
    without raising ``AttributeError: 'numpy.ndarray' object has no attribute 'loc'``.

    The bug was that ``_predict_proba`` converted ``X`` to a numpy array before
    calling ``_get_strata``, but ``_get_strata`` uses pandas ``.loc`` and ``.drop``
    which do not work on numpy arrays.
    """
    statsmodels = pytest.importorskip("statsmodels")  # noqa: F841

    from skpro.survival.coxph import CoxPH

    X, y, C = coxph_strata_data

    # control: unstratified model should work (regression guard)
    control = CoxPH().fit(X[["x"]], y, C=C).predict_proba(X[["x"]])
    assert control.shape == (12, 1), "Unstratified CoxPH predict_proba shape mismatch"

    # stratified model: this was failing before the fix
    model = CoxPH(strata="group").fit(X, y, C=C)
    assert model.is_fitted, "Model should be fitted"

    # this line previously raised:
    # AttributeError: 'numpy.ndarray' object has no attribute 'loc'
    y_pred = model.predict_proba(X)
    assert y_pred.shape == (12, 1), "Stratified CoxPH predict_proba shape mismatch"
