import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from skpro.regression.ondil import OndilOnlineGamlss


@pytest.mark.skipif(
    not _check_soft_dependencies(["ondil"], severity="none"),
    reason="skip test if ondil is not installed in environment",
)
def test_ondil_instantiation_and_get_test_params():
    """Basic smoke test for the Ondil wrapper.

    The test is skipped if the optional dependency ``ondil`` is not
    installed. It verifies that ``get_test_params`` returns at least one
    parameter set and that the estimator can be instantiated with it.
    """
    # ensure ondil import succeeds at runtime; skip the test if import fails
    pytest.importorskip("ondil")

    params = OndilOnlineGamlss.get_test_params()
    if isinstance(params, dict):
        params = [params]
    assert len(params) >= 1

    p = params[0]
    est = OndilOnlineGamlss(**p)
    assert isinstance(est, OndilOnlineGamlss)


@pytest.mark.skipif(
    not _check_soft_dependencies(["ondil"], severity="none"),
    reason="skip test if ondil is not installed in environment",
)
def test_ondil_fit_smoke():
    """Try a light-weight fit call on tiny data to validate wiring.

    This is a smoke test only; if the upstream API requires more complex
    constructor args or data handling, the test will be adjusted later.
    """
    # create tiny dataset
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    y = pd.DataFrame(np.array([[0.1], [1.1], [1.9]]))

    # ensure ondil import succeeds at runtime; skip the test if import fails
    pytest.importorskip("ondil")

    est = OndilOnlineGamlss()

    # fit should run without raising (best-effort); if upstream raises,
    # surface the error so developers can adapt the wrapper.
    est.fit(X, y)
    assert est.is_fitted
