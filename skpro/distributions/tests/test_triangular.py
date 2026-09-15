# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the Triangular distribution parametrization mapping.

The generic distribution suite only checks internal consistency (pdf/cdf/ppf
inversion, log_pdf = log pdf). It cannot detect a wrong ``(lower, mode, upper)``
-> scipy ``(c, loc, scale)`` mapping: a mirrored mapping still yields a valid
triangular that passes every generic test, and it does not check energy
*correctness* (only that a value is produced). These tests pin the mapping and
the hand-derived energy formula to exact closed-form values (cf. the mapping-bug
class of #954 and the Laplace energy bug #720).
"""

__author__ = ["Ashish-Kumar-Dash"]

import numpy as np

from skpro.distributions import Triangular


def test_mapping_semantics():
    """Mean and pdf peak must match the (lower, mode, upper) parametrization.

    Uses an asymmetric triangle, where a mirrored mapping (peak at
    ``lower + upper - mode``) would give a different mean and peak location.
    """
    lower, mode, upper = 0.0, 1.0, 3.0
    mirror = lower + upper - mode  # peak location under a mirrored mapping

    # two identical rows so pdf can be evaluated at mode and mirror at once
    d = Triangular(
        lower=[[lower], [lower]],
        mode=[[mode], [mode]],
        upper=[[upper], [upper]],
    )

    # mean of a triangular is (a + c + b) / 3; a mirror would give 5/3, not 4/3
    assert np.isclose(d.mean().values[0, 0], (lower + mode + upper) / 3)

    # pdf peaks at the mode, not at the mirror point
    pdf = d.pdf(np.array([[mode], [mirror]])).values.flatten()
    assert pdf[0] > pdf[1]


def test_energy_closed_form():
    """Hand-derived energy must match exact closed-form values.

    Self-energy of a triangular is
    ``2(p^2+q^2)/(3d) - 2(p^3+q^3)/(5 d^2)`` with ``d=b-a, p=c-a, q=b-c``;
    for the symmetric case ``p=q`` this is exactly ``7d/30``. The generic
    suite never checks energy against a known value, so pin it here.
    """
    # symmetric triangle on [0, 1]: self-energy = 7/30
    d_sym = Triangular(lower=[[0.0]], mode=[[0.5]], upper=[[1.0]])
    assert np.isclose(d_sym.energy().values[0, 0], 7 / 30)

    # asymmetric triangle (0, 1, 3): 2*5/9 - 2*9/45 = 32/45
    d_asym = Triangular(lower=[[0.0]], mode=[[1.0]], upper=[[3.0]])
    assert np.isclose(d_asym.energy().values[0, 0], 32 / 45)

    # cross-energy at the lower limit equals mean - lower (X >= lower a.s.)
    mean = d_asym.mean().values[0, 0]
    assert np.isclose(d_asym.energy(np.array([[0.0]])).values[0, 0], mean - 0.0)
