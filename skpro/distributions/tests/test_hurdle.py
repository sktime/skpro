import numpy as np
import pytest

from skpro.distributions import Gamma, NegativeBinomial, Normal
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


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize("p", [0.3, np.array([[0.2], [0.7]])])
def test_hurdle_var_is_law_of_total_variance(p):
    """Test var is p * var + p * (1 - p) * mean**2 of the zero-truncated part."""
    inner = Normal(mu=np.array([[3.0, 5.0], [8.0, 10.0]]), sigma=1.5)
    distribution = Hurdle(p=p, distribution=inner)

    truncated = distribution._truncated_distribution
    mean_positive = np.asarray(truncated.mean())
    var_positive = np.asarray(truncated.var())
    p_arr = np.asarray(p)
    expected = p_arr * var_positive + p_arr * (1 - p_arr) * mean_positive**2

    assert np.allclose(np.asarray(distribution.var()), expected)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
def test_hurdle_var_matches_sample_variance():
    """Test var against the variance of a large sample, for a known closed form."""
    distribution = Hurdle(p=0.5, distribution=Normal(mu=10.0, sigma=1.0))

    # zero truncation removes nothing noticeable here: 0.5 * 1 + 0.5 * 0.5 * 10**2
    assert np.isclose(float(np.asarray(distribution.var()).ravel()[0]), 25.5, rtol=1e-3)

    np.random.seed(42)
    samples = distribution.sample(100_000)
    assert np.isclose(float(samples.var().iloc[0]), 25.5, rtol=0.03)


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize(
    "inner",
    [
        Normal(mu=10.0, sigma=1.0),
        Gamma(alpha=2.0, beta=1.0),
        NegativeBinomial(mu=2.0, alpha=1.0),
    ],
    ids=["Normal", "Gamma", "NegativeBinomial"],
)
def test_hurdle_exact_capabilities_follow_zero_truncated_distribution(inner):
    """Test a capability is exact only if the zero-truncated distribution has it."""
    distribution = Hurdle(p=0.5, distribution=inner)
    truncated_exact = distribution._truncated_distribution.get_tag("capabilities:exact")
    declared = Hurdle.get_class_tag("capabilities:exact")

    exact = distribution.get_tag("capabilities:exact")
    approx = distribution.get_tag("capabilities:approx")

    assert set(exact) <= set(truncated_exact)
    # every declared capability is in exactly one of the two lists
    for capa in declared:
        assert (capa in exact) != (capa in approx)
    # nothing the class declared exact is lost, whichever list it ends up in
    assert set(declared) <= set(exact) | set(approx)
