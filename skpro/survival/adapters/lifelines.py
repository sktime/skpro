# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for lifelines models."""

__all__ = ["_LifelinesAdapter"]
__author__ = ["fkiraly"]

from warnings import warn

import numpy as np
import pandas as pd

from skpro.distributions.empirical import Empirical
from skpro.survival.adapters._common import _clip_surv, _get_fitted_params_default_safe
from skpro.utils.sklearn import prep_skl_df


class _LifelinesAdapter:
    """Mixin adapter class for lifelines models."""

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly"],
        "python_dependencies": ["lifelines"],
        "license_type": "permissive",
        # capability tags
        # ---------------
        "X_inner_mtype": "pd_DataFrame_Table",
        "y_inner_mtype": "pd_DataFrame_Table",
        "C_inner_mtype": "pd_DataFrame_Table",
        "capability:multioutput": False,
    }

    # defines the name of the attribute containing the lifelines estimator
    _estimator_attr = "_estimator"

    def _get_lifelines_class(self):
        """Abstract method to get lifelines class.

        should import and return lifelines class
        """
        # from lifelines import LifelinesClass
        #
        # return LifelinesClass
        raise NotImplementedError("abstract method")

    def _get_lifelines_object(self):
        """Abstract method to initialize lifelines object.

        The default initializes result of _get_lifelines_class
        with self.get_params.
        """
        cls = self._get_lifelines_class()
        return cls(**self.get_params())

    def _init_lifelines_object(self):
        """Abstract method to initialize lifelines object and set to _estimator_attr.

        The default writes the return of _get_lifelines_object to
        the attribute of self with name _estimator_attr
        """
        cls = self._get_lifelines_object()
        setattr(self, self._estimator_attr, cls)
        return getattr(self, self._estimator_attr)

    def _get_extra_fit_args(self, X, y, C=None):
        """Get extra arguments for the fit method.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y: pd.DataFrame
            Training labels
        C: pd.DataFrame, optional (default=None)
            Censoring information for survival analysis.

        Returns
        -------
        dict
            Extra arguments for the fit method.
        """
        return {}

    def _fit(self, X, y, C=None):
        """Fit estimator training data.

        Parameters
        ----------
        X : pd.DataFrame
            Training features
        y: pd.DataFrame
            Training labels
        C: pd.DataFrame, optional (default=None)
            Censoring information for survival analysis.

        Returns
        -------
        self: reference to self
            Fitted estimator.
        """
        lifelines_est = self._init_lifelines_object()

        # input conversion
        X = X.astype("float")  # lifelines insists on float dtype
        X = prep_skl_df(X)

        if hasattr(self, "X_col_subset"):
            X = X[self.X_col_subset]

        to_concat = [X, y]

        if C is not None:
            C_col = 1 - C.copy()  # lifelines uses 1 for uncensored, 0 for censored
            C_col.columns = ["__C"]
            to_concat.append(C_col)

        df = pd.concat(to_concat, axis=1)

        self._y_cols = y.columns  # remember column names for later
        y_name = y.columns[0]

        fit_args = {
            "df": df,
            "duration_col": y_name,
        }
        if C is not None:
            fit_args["event_col"] = "__C"

        fit_args.update(self._get_extra_fit_args(X, y, C))

        # fit lifelines estimator
        lifelines_est.fit(**fit_args)

        # write fitted params to self
        # some fitted parameters are properties and may raise exceptions
        # for example, AIC_ and AIC_partial_ of CoxPHFitter
        # to avoid this, we use a safe getter
        lifelines_fitted_params = _get_fitted_params_default_safe(lifelines_est)
        for k, v in lifelines_fitted_params.items():
            setattr(self, f"{k}_", v)

        return self

    def _predict_proba(self, X):
        """Predict_proba method adapter.

        Parameters
        ----------
        X : pd.DataFrame
            Features to predict on.

        Returns
        -------
        skpro Empirical distribution
        """
        lifelines_est = getattr(self, self._estimator_attr)

        # input conversion
        X = X.astype("float")  # lifelines insists on float dtype
        X = prep_skl_df(X)

        # predict on X
        lifelines_survf = lifelines_est.predict_survival_function(X)

        times = lifelines_survf.index

        nt = len(times)
        mi = pd.MultiIndex.from_product([X.index, range(nt)]).swaplevel()

        times_val = np.repeat(times, repeats=len(X))
        times_df = pd.DataFrame(times_val, index=mi, columns=self._y_cols)

        lifelines_survf_t = np.transpose(lifelines_survf.values)
        _, lifelines_survf_t_diff, clipped = _clip_surv(lifelines_survf_t)

        if clipped:
            warn(
                f"Warning from {self.__class__.__name__}: "
                f"Interfaced lifelines class {lifelines_est.__class__.__name__} "
                "produced improper survival function predictions, i.e., "
                "not monotonically decreasing or not in [0, 1]. "
                "skpro has clipped the predictions to enforce proper range and "
                "valid predictive distributions. "
                "However, predictions may still be unreliable.",
                stacklevel=2,
            )

        weights = -lifelines_survf_t_diff.flatten()
        weights_df = pd.Series(weights, index=mi)

        dist = Empirical(
            spl=times_df, weights=weights_df, index=X.index, columns=self._y_cols
        )

        return dist
