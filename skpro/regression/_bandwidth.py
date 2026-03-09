# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Bandwidth selection utilities for 1D Gaussian smoothing bandwidths.

This is a skpro-specific module for regression bandwidth heuristics.
It is not intended for general-purpose use and may have a less stable API
than other skpro modules.

The ISJ and related implementation details are adapted from KDEpy:
https://github.com/tommyod/KDEpy

KDEpy author: Tommy Odland, with contributions from the KDEpy maintainers.
"""

from __future__ import annotations

__author__ = ["joshdunnlime"]

import numpy as np
from scipy.fft import dct
from scipy.optimize import brentq

_BANDWIDTH_METHODS = ("scott", "silverman", "isj")


def _as_1d_float_array(y):
    """Convert input to a finite 1D ``float`` numpy array."""
    y_arr = np.asarray(y, dtype=float)

    if y_arr.ndim == 2 and y_arr.shape[1] == 1:
        y_arr = y_arr[:, 0]
    elif y_arr.ndim != 1:
        raise ValueError("y must be 1D array-like or shape (n, 1).")

    if y_arr.size == 0:
        raise ValueError("y must have at least one element.")

    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y must contain only finite values.")

    return y_arr


def _normalize_weights(weights, n):
    """Return positive normalized weights and the corresponding selection mask.

    Non-positive weights are dropped and therefore excluded from the returned
    aligned sample subset (via the mask).
    """
    if weights is None:
        return None, np.ones(n, dtype=bool)

    w_arr = np.asarray(weights, dtype=float).reshape(-1)

    if w_arr.shape[0] != n:
        raise ValueError("weights must have same length as y.")

    if not np.all(np.isfinite(w_arr)):
        raise ValueError("weights must contain only finite values.")

    mask = w_arr > 0

    if not np.any(mask):
        raise ValueError("weights must contain at least one positive entry.")

    w_arr = w_arr[mask]
    total = w_arr.sum()

    if not np.isfinite(total) or total <= 0:
        raise ValueError("weights must sum to a positive finite number.")

    return w_arr / total, mask


def _effective_sample_size(weights):
    """Compute effective sample size for normalized weights.

    Uses ``n_eff = 1 / sum(w_i^2)``.
    """
    if weights is None:
        return None

    return float(1.0 / np.sum(np.square(weights)))


def _weighted_std(y, weights):
    """Return unweighted/weighted sample scale estimate for a 1D vector.

    For ``weights is None``, uses ``np.std(y, ddof=1)``.
    For normalized weights, uses the weighted second central moment.
    """
    if y.size <= 1:
        return 0.0

    if weights is None:
        return float(np.std(y, ddof=1))

    mean = float(np.sum(weights * y))
    var = float(np.sum(weights * np.square(y - mean)))

    return float(np.sqrt(max(var, 0.0)))


def _resolve_method(method, *, valid):
    """Validate and normalize method name."""
    if not isinstance(method, str):
        raise TypeError("method must be a string.")

    normalized = method.lower()

    if normalized not in valid:
        raise ValueError(f"Unknown method '{method}'. Valid options are {list(valid)}.")

    return normalized


def bw_scott_1d(y, weights=None):
    """Scott 1D bandwidth for Gaussian kernels.

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        1D sample values.
    weights : array-like, optional
        Optional non-negative sample weights.

    Returns
    -------
    float
        Bandwidth interpreted as Gaussian kernel standard deviation.
    """
    y_arr = _as_1d_float_array(y)
    w_norm, mask = _normalize_weights(weights, y_arr.shape[0])
    y_use = y_arr[mask]

    sigma = _weighted_std(y_use, w_norm)
    n_eff = _effective_sample_size(w_norm) if w_norm is not None else y_use.size

    if sigma <= 0 or (n_eff is not None and n_eff <= 1):
        return 0.0

    return float(sigma * (n_eff ** (-1.0 / 5.0)))


def bw_silverman_1d(y, weights=None):
    """Silverman 1D bandwidth for Gaussian kernels.

    Uses the KDEpy-compatible normal-reference formulation in 1D:
    ``h = sigma * (3 n / 4)^(-1/5)``.

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        1D sample values.
    weights : array-like, optional
        Optional non-negative sample weights.

    Returns
    -------
    float
        Bandwidth interpreted as Gaussian kernel standard deviation.
    """
    y_arr = _as_1d_float_array(y)
    w_norm, mask = _normalize_weights(weights, y_arr.shape[0])
    y_use = y_arr[mask]

    n_eff = _effective_sample_size(w_norm) if w_norm is not None else y_use.size

    if n_eff is not None and n_eff <= 1:
        return 0.0

    sigma = _weighted_std(y_use, w_norm)
    if sigma <= 0:
        return 0.0

    return float(sigma * ((3.0 * n_eff / 4.0) ** (-1.0 / 5.0)))


def _autogrid_1d(y, n_grid, boundary_abs, boundary_rel):
    """Create an equidistant padded grid for DCT-based selectors.

    The grid is centered on the data range with additional padding on both sides.

    Parameters
    ----------
    y : array-like, shape (n,)
        1D sample values.
    n_grid : int
        Number of grid points (must be >= 2).
    boundary_abs : float
        Absolute boundary padding added to each side of the data range.
    boundary_rel : float
        Relative boundary padding as a fraction of the data range added to each
        side.

    Returns
    -------
    ndarray, shape (n_grid,)
        Equidistant grid covering the data range with specified padding.
    """
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    data_range = y_max - y_min
    outside = max(float(boundary_rel) * data_range, float(boundary_abs))
    return np.linspace(y_min - outside, y_max + outside, num=int(n_grid), dtype=float)


def _linear_binning_1d(y, grid, weights):
    """Linear 1D binning onto equidistant grid.

    Parameters
    ----------
    y : array-like, shape (n,)
        1D sample values.
    grid : array-like, shape (m,)
        Equidistant grid points (must have length >= 2).
    weights : array-like, shape (n,) or None
        Optional non-negative sample weights. If None, uniform weights are used.

    Returns
    -------
    ndarray, shape (m,)
        Binned probability mass function over grid points (sums to 1 up to
        numerical tolerance).
    """
    n_grid = grid.shape[0]

    if n_grid < 2:
        raise ValueError("grid must have length >= 2.")

    dx = grid[1] - grid[0]

    if not np.isfinite(dx) or dx <= 0:
        raise ValueError("grid must be strictly increasing and finite.")
    if not np.allclose(np.diff(grid), dx, rtol=1e-10, atol=1e-12):
        raise ValueError("grid must be equidistant for DCT-based ISJ.")

    if weights is None:
        w = np.ones_like(y, dtype=float) / y.size
    else:
        w = weights

    t = (y - grid[0]) / dx
    in_grid = (t >= 0.0) & (t <= (n_grid - 1))

    if not np.all(in_grid):
        y = y[in_grid]
        w = w[in_grid]
        t = t[in_grid]

        if y.size == 0:
            raise ValueError("All points were outside the ISJ grid.")

    idx = np.floor(t).astype(int)
    idx = np.minimum(idx, n_grid - 2)
    frac = t - idx

    pmf = np.zeros(n_grid, dtype=float)
    np.add.at(pmf, idx, w * (1.0 - frac))
    np.add.at(pmf, idx + 1, w * frac)

    total = pmf.sum()

    if total > 0:
        pmf /= total

    return pmf


def _isj_fixed_point(t, n_eff, i_sq, a2):
    """Fixed-point function for the ISJ diffusion estimator.

    Parameters
    ----------
    t : float
        Diffusion time parameter.
    n_eff : float
        Effective sample size.
    i_sq : array-like
        Squared Discrete Cosine Transform indices.
    a2 : array-like
        Squared Discrete Cosine Transform coefficients.

    Returns
    -------
    float
        Difference between t and the ISJ fixed-point solution for given parameters.
    """
    float_type = np.longdouble
    i_sq = np.asarray(i_sq, dtype=float_type)
    a2 = np.asarray(a2, dtype=float_type)

    ell = 7

    f_val = (
        0.5
        * (np.pi ** (2 * ell))
        * np.sum((i_sq**ell) * a2 * np.exp(-(i_sq * (np.pi**2) * t)))
    )

    if f_val <= 0:
        return -1.0

    for s in range(ell - 1, 1, -1):
        odd_prod = np.prod(np.arange(1, 2 * s + 1, 2, dtype=float_type))
        k0 = odd_prod / np.sqrt(2 * np.pi)
        const = (1.0 + (0.5 ** (s + 0.5))) / 3.0
        t_s = (2.0 * const * k0 / (n_eff * f_val)) ** (2.0 / (3.0 + 2.0 * s))

        f_val = (
            0.5
            * (np.pi ** (2 * s))
            * np.sum((i_sq**s) * a2 * np.exp(-(i_sq * (np.pi**2) * t_s)))
        )

    t_opt = (2.0 * n_eff * np.sqrt(np.pi) * f_val) ** (-2.0 / 5.0)

    return float(t - t_opt)


def _isj_root(n_eff, i_sq, a2):
    """Find positive root for ISJ fixed-point equation.

    Parameters
    ----------
    n_eff : float
        Effective sample size.
    i_sq : array-like
        Squared Discrete Cosine Transform indices.
    a2 : array-like
        Squared Discrete Cosine Transform coefficients.

    Returns
    -------
    float
        Positive root of the ISJ fixed-point equation.
    """
    n_eff = int(max(50, min(1050, n_eff)))
    upper = 1e-11 + 0.01 * (n_eff - 50) / 1000.0
    max_upper = 1.0

    while upper < max_upper:
        try:
            root, res = brentq(
                lambda t: _isj_fixed_point(t, n_eff=n_eff, i_sq=i_sq, a2=a2),
                0.0,
                upper,
                full_output=True,
                disp=False,
            )

            if res.converged and root > 0:
                return float(root)

        except ValueError:
            pass

        upper *= 2.0

    raise ValueError("ISJ root finding did not converge.")


def bw_isj_1d(
    y,
    weights=None,
    *,
    n_grid=2**10,
    boundary_abs=6.0,
    boundary_rel=0.5,
    fallback="silverman",
):
    """Improved Sheather-Jones bandwidth for 1D Gaussian kernels.

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        1D sample values.
    weights : array-like, optional
        Optional non-negative sample weights.
    n_grid : int, optional
        Number of grid points for DCT (power of two and >= 16 for stability).
        Default is 1024.
    boundary_abs : float, optional
        Absolute boundary padding added to each side of the data range for the
        DCT grid. Default is 6.0.
    boundary_rel : float, optional
        Relative boundary padding as a fraction of the data range added to each
        side of the DCT grid. Default is 0.5.
    fallback : {"silverman", "scott", "none"}, optional
        Fallback method if ISJ root finding fails. Default is "silverman". If
        "none", an exception is raised instead. Used only if ISJ fails to
        converge.

    Returns
    -------
    float
        Bandwidth interpreted as Gaussian kernel standard deviation.

    Notes
    -----
    If the data are degenerate (zero range), returns ``0.0`` directly.
    """
    y_arr = _as_1d_float_array(y)
    w_norm, mask = _normalize_weights(weights, y_arr.shape[0])
    y_use = y_arr[mask]

    y_min = float(np.min(y_use))
    y_max = float(np.max(y_use))
    data_range = y_max - y_min

    if not np.isfinite(data_range) or data_range <= 0:
        return 0.0

    if int(n_grid) < 16 or int(n_grid) & (int(n_grid) - 1):
        raise ValueError("n_grid must be a power of two and >= 16 for ISJ stability.")

    grid = _autogrid_1d(
        y_use, n_grid=int(n_grid), boundary_abs=boundary_abs, boundary_rel=boundary_rel
    )

    pmf = _linear_binning_1d(y_use, grid=grid, weights=w_norm)
    coeffs = dct(pmf, type=2, norm=None)

    i_sq = np.arange(1, int(n_grid), dtype=np.longdouble) ** 2
    a2 = np.asarray(coeffs[1:], dtype=np.longdouble) ** 2

    n_eff = _effective_sample_size(w_norm) if w_norm is not None else y_use.size

    try:
        t_star = _isj_root(n_eff=max(2, int(round(n_eff))), i_sq=i_sq, a2=a2)
        return float(np.sqrt(t_star) * data_range)
    except Exception as exc:
        fallback = _resolve_method(fallback, valid=("silverman", "scott", "none"))

        if fallback == "none":
            raise ValueError("ISJ bandwidth estimation failed to converge.") from exc

        return bandwidth_1d(y_use, method=fallback, weights=w_norm)


def bandwidth_1d(y, method="silverman", weights=None, **kwargs):
    """Select 1D bandwidth for Gaussian smoothing.

    Parameters
    ----------
    y : array-like, shape (n,) or (n, 1)
        1D sample values.
    method : {"silverman", "scott", "isj"}
        Bandwidth heuristic.
    weights : array-like, optional
        Optional non-negative sample weights.
    **kwargs : dict
        Additional kwargs passed to ``bw_isj_1d`` for ``method="isj"``.

    Returns
    -------
    float
        Bandwidth interpreted as Gaussian kernel standard deviation.
    """
    method = _resolve_method(method, valid=_BANDWIDTH_METHODS)

    if method == "silverman":
        return bw_silverman_1d(y, weights=weights)

    if method == "scott":
        return bw_scott_1d(y, weights=weights)

    return bw_isj_1d(y, weights=weights, **kwargs)


__all__ = [
    "bandwidth_1d",
    "bw_isj_1d",
    "bw_scott_1d",
    "bw_silverman_1d",
]
