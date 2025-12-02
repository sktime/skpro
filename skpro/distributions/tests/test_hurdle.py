import numpy as np
import numpy.testing as npt
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


@pytest.mark.skipif(
    not run_test_module_changed("skpro.distributions"),
    reason="run only if skpro.distributions has been changed",
)
@pytest.mark.parametrize("params", Hurdle.get_test_params())
def test_hurdle_regression(params):
    """Regression test for Hurdle distribution methods."""
    dist = Hurdle(**params)

    xs = np.array([-1.0, 0.0, 1.0, 3.0])

    pmf_vals = np.asarray(dist.pmf(xs))
    log_pmf_vals = np.asarray(dist.log_pmf(xs))
    pdf_vals = np.asarray(dist.pdf(xs))
    log_pdf_vals = np.asarray(dist.log_pdf(xs))
    cdf_vals = np.asarray(dist.cdf(xs))

    p = np.asarray(dist.p)
    if p.size == 1:
        p_b = np.full(xs.shape, p.item())
    else:
        try:
            p_b = np.broadcast_to(p, xs.shape)
        except Exception:
            p_b = np.asarray(p)

    trunc = dist._truncated_distribution

    neg_mask = xs < 0.0
    if neg_mask.any():
        npt.assert_array_equal(pmf_vals[neg_mask], 0.0)
        npt.assert_array_equal(pdf_vals[neg_mask], 0.0)
        npt.assert_array_equal(cdf_vals[neg_mask], 0.0)
        assert np.all(np.isneginf(log_pmf_vals[neg_mask]))
        assert np.all(np.isneginf(log_pdf_vals[neg_mask]))

    zero_mask = xs == 0.0
    if zero_mask.any():
        p_zero = p_b[zero_mask] if p_b.shape == xs.shape else (1 - p)
        npt.assert_allclose(pmf_vals[zero_mask], (1.0 - p_zero))
        npt.assert_allclose(cdf_vals[zero_mask], (1.0 - p_zero))
        npt.assert_allclose(log_pmf_vals[zero_mask], np.log(1.0 - p_zero))

    pos_mask = xs > 0.0
    if pos_mask.any():
        xs_pos = xs[pos_mask]

        trunc_pmf = np.asarray(trunc.pmf(xs_pos))
        trunc_pdf = np.asarray(trunc.pdf(xs_pos))
        trunc_log_pmf = np.asarray(trunc.log_pmf(xs_pos))
        trunc_log_pdf = np.asarray(trunc.log_pdf(xs_pos))
        trunc_cdf = np.asarray(trunc.cdf(xs_pos))

        p_pos = p_b[pos_mask] if p_b.shape == xs.shape else p

        npt.assert_allclose(pmf_vals[pos_mask], p_pos * trunc_pmf)
        npt.assert_allclose(pdf_vals[pos_mask], p_pos * trunc_pdf)

        npt.assert_allclose(log_pmf_vals[pos_mask], np.log(p_pos) + trunc_log_pmf)
        npt.assert_allclose(log_pdf_vals[pos_mask], np.log(p_pos) + trunc_log_pdf)

        npt.assert_allclose(cdf_vals[pos_mask], (1.0 - p_pos) + p_pos * trunc_cdf)
