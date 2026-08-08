"""Maximum likelihood distribution fitter via scipy."""
# copyright: skpro developers, BSD-3-Clause License (see LICENSE file)

from skpro.distfitter.base import BaseDistFitter

_DEFAULT_PARAM_MAPS = {
    "Alpha": {"a": ("a", "identity")},
    "Beta": {"alpha": ("a", "identity"), "beta": ("b", "identity")},
    "Cauchy": {"mu": ("loc", "identity"), "scale": ("scale", "identity")},
    "ChiSquared": {"dof": ("df", "identity")},
    "Erlang": {"rate": ("scale", "inverse"), "k": ("a", "identity")},
    "Exponential": {"rate": ("scale", "inverse")},
    "FatigueLife": {"c": ("c", "identity"), "scale": ("scale", "identity")},
    "Fisk": {"alpha": ("scale", "identity"), "beta": ("c", "identity")},
    "Gamma": {"alpha": ("a", "identity"), "beta": ("scale", "inverse")},
    "GeneralizedPareto": {
        "c": ("c", "identity"),
        "mu": ("loc", "identity"),
        "scale": ("scale", "identity"),
    },
    "Gompertz": {"c": ("c", "identity"), "scale": ("scale", "identity")},
    "GumbelL": {"mu": ("loc", "identity"), "sigma": ("scale", "identity")},
    "GumbelR": {"mu": ("loc", "identity"), "sigma": ("scale", "identity")},
    "HalfCauchy": {"beta": ("scale", "identity")},
    "HalfLogistic": {"beta": ("scale", "identity")},
    "HalfNormal": {"sigma": ("scale", "identity")},
    "InverseGamma": {"alpha": ("a", "identity"), "beta": ("scale", "identity")},
    "InverseGaussian": {"mu": ("mu", "identity"), "scale": ("scale", "identity")},
    "Laplace": {"mu": ("loc", "identity"), "scale": ("scale", "identity")},
    "Levy": {"mu": ("loc", "identity"), "scale": ("scale", "identity")},
    "LogGamma": {"c": ("c", "identity")},
    "Normal": {"mu": ("loc", "identity"), "sigma": ("scale", "identity")},
    "SkewNormal": {
        "mu": ("loc", "identity"),
        "sigma": ("scale", "identity"),
        "alpha": ("a", "identity"),
    },
    "TDistribution": {
        "mu": ("loc", "identity"),
        "sigma": ("scale", "identity"),
        "df": ("df", "identity"),
    },
}

_DEFAULT_FIT_KWARGS = {
    "Alpha": {"floc": 0},
    "ChiSquared": {"floc": 0, "fscale": 1},
    "Erlang": {"floc": 0},
    "Exponential": {"floc": 0},
    "FatigueLife": {"floc": 0},
    "Fisk": {"floc": 0},
    "Gamma": {"floc": 0},
    "Gompertz": {"floc": 0},
    "HalfCauchy": {"floc": 0},
    "HalfLogistic": {"floc": 0},
    "HalfNormal": {"floc": 0},
    "InverseGamma": {"floc": 0},
    "InverseGaussian": {"floc": 0},
    "LogGamma": {"floc": 0},
}

_DEFAULT_SCIPY_DIST = {
    "ChiSquared": "chi2",
    "Laplace": "laplace",
    "Normal": "norm",
    "TDistribution": "t",
}


class ScipyMLEFitter(BaseDistFitter):
    r"""Fit a distribution by maximum likelihood estimation via scipy.

    Uses ``scipy.stats.<dist>.fit(data)`` to obtain MLE parameter estimates,
    then maps the fitted scipy parameters back to skpro distribution
    constructor arguments using ``param_map``.

    Scipy's ``fit`` method returns a tuple
    ``(*shape_params, loc, scale)`` where shape parameter names are given
    by ``scipy_dist.shapes``.  The ``param_map`` dictionary translates
    each scipy result element to the corresponding skpro constructor
    argument, optionally applying a simple transform.

    When ``param_map`` is not provided, built-in defaults are used for the
    following distributions:

    * ``Alpha``, ``Beta``, ``Cauchy``, ``ChiSquared``, ``Erlang``,
      ``Exponential``, ``FatigueLife``, ``Fisk``, ``Gamma``,
      ``GeneralizedPareto``, ``Gompertz``, ``GumbelL``, ``GumbelR``,
      ``HalfCauchy``, ``HalfLogistic``, ``HalfNormal``, ``InverseGamma``,
      ``InverseGaussian``, ``Laplace``, ``Levy``, ``LogGamma``, ``Normal``,
      ``SkewNormal``, ``TDistribution``.

    Sensible ``fit_kwargs`` (e.g. ``floc=0``) and ``scipy_dist`` are also
    supplied automatically for the distributions listed above when the
    respective arguments are left at their default ``None``.

    Parameters
    ----------
    dist : skpro distribution class
        A distribution class from ``skpro.distributions``.
        Must be a class, not an instance.
    param_map : dict or None, optional (default=None)
        Mapping from skpro parameter names to ``(scipy_key, transform)``
        pairs.  When ``None``, uses built-in defaults for supported
        distributions (see list above).

        - ``scipy_key`` is the name of the scipy parameter in the fit
          result. It can be one of the shape parameter names from
          ``scipy_dist.shapes``, or the strings ``"loc"`` and ``"scale"``.
        - ``transform`` is one of:

          - ``"identity"`` : use the scipy value directly.
          - ``"inverse"`` : use ``1 / scipy_value``.

        Example for ``Exponential(rate=...)``:
        ``{"rate": ("scale", "inverse")}``

        Example for ``Gamma(alpha=..., beta=...)``:
        ``{"alpha": ("a", "identity"), "beta": ("scale", "inverse")}``
    scipy_dist : scipy.stats distribution or None, optional (default=None)
        The scipy distribution object whose ``.fit()`` method will be
        called.  When ``None``, auto-detected from the skpro ``dist``
        class (works for ``_ScipyAdapter`` subclasses and for supported
        distributions listed above).
    fit_kwargs : dict or None, optional (default=None)
        Additional keyword arguments passed to ``scipy.stats.<dist>.fit``.
        When ``None``, sensible defaults are applied for supported
        distributions (e.g. ``{"floc": 0}`` for distributions with
        non-negative support).  Pass an empty dict ``{}`` to suppress
        automatic defaults.

    Examples
    --------
    Using built-in defaults (no ``param_map`` needed):

    >>> import pandas as pd
    >>> from skpro.distfitter import ScipyMLEFitter
    >>> from skpro.distributions.exponential import Exponential
    >>> X = pd.DataFrame([0.5, 1.0, 1.5, 2.0, 2.5])
    >>> fitter = ScipyMLEFitter(dist=Exponential)
    >>> fitter.fit(X)
    ScipyMLEFitter(...)
    >>> dist = fitter.proba()

    With explicit ``param_map``:

    >>> from skpro.distributions.normal import Normal
    >>> fitter = ScipyMLEFitter(
    ...     dist=Normal,
    ...     param_map={"mu": ("loc", "identity"), "sigma": ("scale", "identity")},
    ... )
    >>> fitter.fit(X)
    ScipyMLEFitter(...)
    """

    _tags = {
        "authors": ["patelchaitany"],
        "reserved_params": ["dist"],
    }

    def __init__(self, dist, param_map=None, scipy_dist=None, fit_kwargs=None):
        self.dist = dist
        self.param_map = param_map
        self.scipy_dist = scipy_dist
        self.fit_kwargs = fit_kwargs

        super().__init__()

    def _resolve_defaults(self):
        """Resolve param_map, fit_kwargs, and scipy_dist from built-in defaults."""
        dist_name = self.dist.__name__

        param_map = self.param_map
        if param_map is None:
            if dist_name not in _DEFAULT_PARAM_MAPS:
                raise ValueError(
                    f"No built-in param_map for {dist_name!r}. "
                    f"Pass param_map explicitly. "
                    f"Supported: {sorted(_DEFAULT_PARAM_MAPS.keys())}."
                )
            param_map = _DEFAULT_PARAM_MAPS[dist_name]

        fit_kwargs = self.fit_kwargs
        if fit_kwargs is None:
            fit_kwargs = _DEFAULT_FIT_KWARGS.get(dist_name, {})

        scipy_obj = self.scipy_dist
        if scipy_obj is None:
            if dist_name in _DEFAULT_SCIPY_DIST:
                import scipy.stats

                scipy_obj = getattr(scipy.stats, _DEFAULT_SCIPY_DIST[dist_name])
            else:
                scipy_obj = self._get_scipy_obj_from_adapter()

        return param_map, fit_kwargs, scipy_obj

    def _fit(self, X, C=None):
        """Fit distribution parameters by maximum likelihood via scipy.

        Parameters
        ----------
        X : pandas DataFrame
            Data to fit the distribution to.
        C : ignored

        Returns
        -------
        self : reference to self
        """
        vals = X.values.ravel()

        param_map, fit_kwargs, scipy_obj = self._resolve_defaults()
        fit_result = scipy_obj.fit(vals, **fit_kwargs)

        scipy_result_dict = self._fit_result_to_dict(scipy_obj, fit_result)

        self.fitted_params_ = self._apply_param_map(scipy_result_dict, param_map)
        return self

    def _get_scipy_obj_from_adapter(self):
        """Extract scipy object from a _ScipyAdapter subclass."""
        from skpro.distributions.adapters.scipy import _ScipyAdapter

        if isinstance(self.dist, type) and issubclass(self.dist, _ScipyAdapter):
            dummy = object.__new__(self.dist)
            return dummy._get_scipy_object()

        raise TypeError(
            f"Cannot auto-detect scipy distribution from {self.dist!r}. "
            f"Pass scipy_dist explicitly."
        )

    @staticmethod
    def _fit_result_to_dict(scipy_obj, fit_result):
        """Convert scipy fit result tuple to a named dict.

        Parameters
        ----------
        scipy_obj : scipy.stats distribution
        fit_result : tuple from scipy_obj.fit()

        Returns
        -------
        dict : {param_name: value}
            Keys are shape parameter names plus "loc" and "scale".
        """
        shapes_str = scipy_obj.shapes
        if shapes_str is not None:
            shape_names = [s.strip() for s in shapes_str.split(",")]
        else:
            shape_names = []

        result = {}
        for i, name in enumerate(shape_names):
            result[name] = fit_result[i]

        n_shapes = len(shape_names)
        result["loc"] = fit_result[n_shapes]
        result["scale"] = fit_result[n_shapes + 1]

        return result

    @staticmethod
    def _apply_param_map(scipy_result_dict, param_map):
        """Apply param_map to convert scipy params to skpro params.

        Parameters
        ----------
        scipy_result_dict : dict
            Named scipy fit results.
        param_map : dict
            Mapping from skpro param names to (scipy_key, transform).

        Returns
        -------
        dict : {skpro_param: value}
        """
        TRANSFORMS = {
            "identity": lambda x: x,
            "inverse": lambda x: 1.0 / x,
        }

        skpro_params = {}
        for skpro_name, (scipy_key, transform) in param_map.items():
            if scipy_key not in scipy_result_dict:
                raise ValueError(
                    f"scipy_key {scipy_key!r} not found in fit result. "
                    f"Available keys: {list(scipy_result_dict.keys())}"
                )
            if transform not in TRANSFORMS:
                raise ValueError(
                    f"Unknown transform {transform!r}. "
                    f"Must be one of {list(TRANSFORMS.keys())}."
                )

            raw = float(scipy_result_dict[scipy_key])
            skpro_params[skpro_name] = TRANSFORMS[transform](raw)

        return skpro_params

    def _proba(self):
        """Return fitted distribution.

        Returns
        -------
        dist : skpro BaseDistribution (scalar)
        """
        return self.dist(**self.fitted_params_)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        from skpro.distributions.exponential import Exponential
        from skpro.distributions.normal import Normal

        params1 = {"dist": Normal}
        params2 = {"dist": Exponential}
        return [params1, params2]
