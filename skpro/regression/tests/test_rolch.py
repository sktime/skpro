import pytest
import pandas as pd
import numpy as np

try:
    import ondil  # noqa: F401
except Exception:
    ondil = None

from skpro.regression.rolch import RolchOnlineGamlss


if ondil is None:
    pytest.skip("ondil not installed", allow_module_level=True)


def test_rolch_instantiation_and_get_test_params():
    """Basic smoke test for the Rolch wrapper.

    The test is skipped if the optional dependency ``ondil`` is not
    installed. It verifies that ``get_test_params`` returns at least one
    parameter set and that the estimator can be instantiated with it.
    """
    params = RolchOnlineGamlss.get_test_params()
    if isinstance(params, dict):
        params = [params]
    assert len(params) >= 1

    p = params[0]
    est = RolchOnlineGamlss(**p)
    assert isinstance(est, RolchOnlineGamlss)


def test_rolch_fit_smoke():
    """Try a light-weight fit call on tiny data to validate wiring.

    This is a smoke test only; if the upstream API requires more complex
    constructor args or data handling, the test will be adjusted later.
    """
    # create tiny dataset
    X = pd.DataFrame({"a": [0.0, 1.0, 2.0]})
    y = pd.DataFrame(np.array([[0.1], [1.1], [1.9]]))

    est = RolchOnlineGamlss()

    # fit should run without raising (best-effort); if upstream raises,
    # surface the error so developers can adapt the wrapper.
    est.fit(X, y)
    assert est.is_fitted
