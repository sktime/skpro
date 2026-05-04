"""Interface for ondil OnlineGamlss probabilistic regressor.

This module provides a lightweight wrapper around ``ondil``'s
``OnlineGamlss`` estimator to expose it as an skpro ``BaseProbaRegressor``.

The wrapper is intentionally lightweight: imports of the optional
``ondil`` dependency are performed inside methods so the package is
optional for users who do not need this estimator.

The wrapper attempts to be tolerant to different method names used by
the upstream estimator: it will use ``fit`` when available, otherwise
fall back to ``partial_fit`` or ``update`` where appropriate. Prediction
is best-effort: if the upstream ``predict`` method returns distribution
parameters (e.g., columns for location/scale) these are converted to a
``skpro.distributions`` object; otherwise an informative error is raised.
"""

from skpro.regression.base import BaseProbaRegressor


class OndilOnlineGamlss(BaseProbaRegressor):
    """Wrapper for ondil.online_gamlss.OnlineGamlss.

    Parameters
    ----------
    distribution : str, default="Normal"
        Name of distribution to expose via skpro. This is used to map
        parameter names returned by the upstream estimator to skpro's
        distribution constructors. Common value is "Normal".

    Notes
    -----
        * The ondil dependency is optional and imported inside methods.
        * The wrapper uses a best-effort strategy to call the appropriate
            fit/update/predict methods of the upstream estimator. If ondil's
            API changes in incompatible ways, this wrapper may need updates.
    """

    _tags = {
        "authors": ["arnavk23"],
        "maintainers": ["fkiraly"],
        "python_dependencies": ["ondil"],
        "capability:multioutput": False,
        "capability:missing": True,
        "tests:vm": True,
        "capability:update": True,
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
    }

    def __init__(self, distribution="Normal", ondil_init_params=None):
        """Initialize OndilOnlineGamlss.

        Parameters
        ----------
        distribution : str, default="Normal"
            Name of distribution to expose via skpro.
        ondil_init_params : dict, optional
            Parameters to forward to ondil's OnlineGamlss constructor.
        """
        self.distribution = distribution
        self.ondil_init_params = ondil_init_params
        # explicit dict of kwargs forwarded to the ondil constructor.
        self._ondil_kwargs = dict(ondil_init_params or {})

        super().__init__()

    def _fit(self, X, y):
        """Fit the underlying ondil OnlineGamlss estimator.

        The method tries several common fitting/update method names in the
        upstream estimator to support different ondil versions.
        """
        # defer import to keep ondil optional
        import importlib

        module_str = "ondil.estimators.online_gamlss"
        ondil_mod = importlib.import_module(module_str)
        try:
            OnlineGamlss = ondil_mod.OnlineDistributionalRegression
        except AttributeError:
            try:
                OnlineGamlss = ondil_mod.OnlineGamlss
            except AttributeError as exc:
                raise ImportError(
                    "ondil.estimators.online_gamlss does not expose "
                    "'OnlineDistributionalRegression' or 'OnlineGamlss' - "
                    "please install a compatible ondil version"
                ) from exc

        # ensure DataFrame column names are strings to satisfy upstream
        # validation that may require string feature names
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = X.columns.astype(str)
        if isinstance(y, pd.DataFrame):
            y = y.copy()
            y.columns = y.columns.astype(str)

        # store y columns for later (stringified)
        self._y_cols = y.columns

        # instantiate upstream estimator
        self._ondil = OnlineGamlss(**self._ondil_kwargs)

        # Prefer `fit`, then `partial_fit`, then `update`.
        for method_name in ("fit", "partial_fit", "update"):
            if hasattr(self._ondil, method_name):
                method = getattr(self._ondil, method_name)
                # Call the upstream method with the provided X, y.
                method(X, y)
                break
        else:
            raise AttributeError(
                "ondil OnlineGamlss instance has no fit/partial_fit/update method"
            )

        return self

    def _update(self, X, y):
        """Update the fitted ondil estimator in online fashion.

        Tries common update method names on the upstream estimator.
        """
        if not hasattr(self, "_ondil"):
            raise RuntimeError("Estimator not fitted yet; call fit before update")

        import pandas as pd

        # sanitize column names before delegating to upstream methods
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = X.columns.astype(str)
        if isinstance(y, pd.DataFrame):
            y = y.copy()
            y.columns = y.columns.astype(str)

        if hasattr(self._ondil, "update"):
            self._ondil.update(X, y)
            return self

        if hasattr(self._ondil, "partial_fit"):
            self._ondil.partial_fit(X, y)
            return self

        raise AttributeError(
            "Upstream ondil estimator has no update/partial_fit method"
        )

    def _predict_proba(self, X):
        """Predict distribution parameters and convert to skpro distribution.

        The method is best-effort: it tries to call ``predict`` on the
        underlying ondil estimator and expects a pandas DataFrame (or array)
        of parameters. For the common case of a Normal prediction the
        columns should contain location and scale (names tolerated below).
        """
        import importlib

        import pandas as pd

        if not hasattr(self, "_ondil"):
            raise RuntimeError("Estimator not fitted yet; call fit before predict")

        # call predict on upstream estimator

        # ensure feature names are strings for upstream validation
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = X.columns.astype(str)

        if hasattr(self._ondil, "predict"):
            params = self._ondil.predict(X)
        elif hasattr(self._ondil, "predict_params"):
            params = self._ondil.predict_params(X)
        else:
            raise AttributeError("Upstream ondil estimator has no predict method")

        # normalize to pandas DataFrame
        if isinstance(params, pd.DataFrame):
            df = params
        else:
            try:
                df = pd.DataFrame(params)
            except Exception as e:  # noqa: B902
                raise TypeError("Unrecognized predict output from ondil: %s" % e)

        # decide mapping based on requested distribution
        dist = self.distribution
        # import skpro distributions lazily
        distr_mod = importlib.import_module("skpro.distributions")

        if dist == "Normal":
            # accept common column names for loc/scale
            col_candidates = {
                "loc": ["loc", "mu", "mean"],
                "scale": ["scale", "sigma", "sd", "std"],
            }

            def _find(col_names):
                for c in col_names:
                    if c in df.columns:
                        return c
                return None

            loc_col = _find(col_candidates["loc"]) or df.columns[0]

            # Check if we have a scale column
            scale_col = _find(col_candidates["scale"])

            # If no explicit scale column found, check if we have multiple columns
            if scale_col is None and df.shape[1] > 1:
                # Use second column as scale
                scale_col = df.columns[1]

            loc = df.loc[:, [loc_col]].values

            if scale_col is not None:
                scale = df.loc[:, [scale_col]].values
            else:
                # If ondil only returns one column (location), use a default scale
                # This is common with default GAMLSS configurations that only estimate
                # the mean. We use a scale of 1.0 (constant for all predictions).
                # Note: This may not be ideal and may underestimate uncertainty.
                import numpy as np

                scale = np.ones((len(df), 1))

            Normal = distr_mod.Normal
            return Normal(mu=loc, sigma=scale, index=X.index, columns=self._y_cols)

        # fallback: try to call distribution class with all columns as kwargs
        if hasattr(distr_mod, dist):
            Distr = getattr(distr_mod, dist)
            vals = {str(c): df.loc[:, [c]].values for c in df.columns}
            return Distr(**vals, index=X.index, columns=self._y_cols)

        raise NotImplementedError(
            "Mapping to skpro distribution '" + str(dist) + "' not implemented"
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        The ondil dependency is optional: the test harness will skip tests
        requiring ondil if the package is not available on the test runner.
        """
        # minimal constructor params; provide two small parameter sets so
        # the package-level tests exercise different constructor paths.
        return [
            {"distribution": "Normal"},
            {"distribution": "Normal", "ondil_init_params": {"verbose": 0}},
        ]
