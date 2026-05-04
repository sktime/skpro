import numpy as np
import pytest

from skpro.distributions.hurdle import Hurdle
from skpro.tests.test_switch import run_test_module_changed


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize("params", Hurdle.get_test_params())
def test_hurdle_less_than_zero(params):
    """Test that the index is correctly set after iat call."""
    distribution = Hurdle(**params)

    v = -1.0

    funcs_and_expected = [
        (distribution.cdf, 0.0),
        (distribution.pdf, 0.0),
        (distribution.pmf, 0.0),
        (distribution.log_pdf, -np.inf),
        (distribution.log_pmf, -np.inf),
    ]

    for func, expected in funcs_and_expected:
        values = func(v)
        assert (np.asarray(values) == expected).all()
