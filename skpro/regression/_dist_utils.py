# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)
"""Distribution string normalisation utility for skpro regressors.

Provides ``_normalize_dist_str``, which maps all known string aliases for a
probability distribution to the canonical capitalized class name used in skpro
(e.g. ``"gaussian"`` -> ``"Normal"``, ``"t"`` -> ``"TDistribution"``).

Every probabilistic regressor adapter should call this before its own internal
string -> object mapping so that users can pass any reasonable alias and have it
work uniformly across regressors and in GridSearchCV / RandomizedSearchCV.
"""

_DIST_ALIAS_MAP: dict[str, str] = {
    # Normal / Gaussian
    "normal": "Normal",
    "gaussian": "Normal",
    "norm": "Normal",
    # Laplace
    "laplace": "Laplace",
    "double_exponential": "Laplace",
    # LogNormal
    "lognormal": "LogNormal",
    "log_normal": "LogNormal",
    "log-normal": "LogNormal",
    "log normal": "LogNormal",
    # TDistribution
    "tdistribution": "TDistribution",
    "t_distribution": "TDistribution",
    "t-distribution": "TDistribution",
    "t": "TDistribution",
    "student_t": "TDistribution",
    "studentt": "TDistribution",
    "student-t": "TDistribution",
    # Poisson
    "poisson": "Poisson",
    # Exponential
    "exponential": "Exponential",
    "exp": "Exponential",
    # Gamma
    "gamma": "Gamma",
    # Beta
    "beta": "Beta",
    # Weibull
    "weibull": "Weibull",
    # Cauchy
    "cauchy": "Cauchy",
    # Binomial
    "binomial": "Binomial",
    "binom": "Binomial",
    # NegativeBinomial
    "negativebinomial": "NegativeBinomial",
    "negative_binomial": "NegativeBinomial",
    "negative.binomial": "NegativeBinomial",
    "negbinomial": "NegativeBinomial",
    "negbin": "NegativeBinomial",
    "neg_binomial": "NegativeBinomial",
    # InverseGaussian
    "inversegaussian": "InverseGaussian",
    "inverse_gaussian": "InverseGaussian",
    "inverse.gaussian": "InverseGaussian",
    "inv_gaussian": "InverseGaussian",
    # Tweedie
    "tweedie": "Tweedie",
    # Logistic / SinhLogistic (QPD inner distributions used by CyclicBoosting)
    "logistic": "Logistic",
    "sinhlogistic": "SinhLogistic",
    "sinh_logistic": "SinhLogistic",
    "sinh-logistic": "SinhLogistic",
}


def _normalize_dist_str(dist: str) -> str:
    """Normalize a distribution string to the canonical capitalized class name.

    Maps every known alias (case-insensitive) to the capitalized skpro class
    name, e.g.::

        _normalize_dist_str("gaussian")   -> "Normal"
        _normalize_dist_str("t")          -> "TDistribution"
        _normalize_dist_str("lognormal")  -> "LogNormal"
        _normalize_dist_str("Normal")     -> "Normal"  # already canonical

    Non-string inputs (e.g. a distribution class or object) are returned
    unchanged so callers do not need to guard separately.

    Unknown strings emit a ``UserWarning`` and are returned as-is to preserve
    backward-compatibility with any existing library-specific aliases.

    Parameters
    ----------
    dist : str
        Distribution name in any accepted format.

    Returns
    -------
    str
        Canonical distribution name (capitalised class name in skpro).

    Examples
    --------
    >>> from skpro.regression._dist_utils import _normalize_dist_str
    >>> _normalize_dist_str("gaussian")
    'Normal'
    >>> _normalize_dist_str("lognormal")
    'LogNormal'
    >>> _normalize_dist_str("t")
    'TDistribution'
    >>> _normalize_dist_str("Normal")
    'Normal'
    """
    if not isinstance(dist, str):
        return dist

    lower = dist.lower()

    # 1. Direct alias lookup (handles the vast majority of cases)
    if lower in _DIST_ALIAS_MAP:
        return _DIST_ALIAS_MAP[lower]

    # 2. Unknown — warn but do not raise (preserves backward-compatibility)
    import warnings

    warnings.warn(
        f"Distribution string '{dist}' is not recognised by _normalize_dist_str "
        f"and will be passed through unchanged. If this is intentional, consider "
        f"adding it to _DIST_ALIAS_MAP in skpro/regression/_dist_utils.py.",
        UserWarning,
        stacklevel=2,
    )
    return dist
